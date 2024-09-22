from pathlib import Path
from typing import Any, Optional, Union, Callable

import pytorch_lightning as pl
import torch
from diffusers import DDPMScheduler, DiffusionPipeline, AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat

from transformers import CLIPTextModel, CLIPTokenizer
from utils.video_utils import ResultProcessor, save_videos_grid, video_naming

from t2v_enhanced.model import pl_module_params_controlnet

from t2v_enhanced.model.diffusers_conditional.models.controlnet.controlnet import ControlNetModel
from t2v_enhanced.model.diffusers_conditional.models.controlnet.unet_3d_condition import UNet3DConditionModel
from t2v_enhanced.model.diffusers_conditional.models.controlnet.pipeline_text_to_video_w_controlnet_synth import TextToVideoSDPipeline
from t2v_enhanced.model.diffusers_conditional.models.controlnet.pipeline_text_to_video_synth import TextToVideoSDPipeline as TextToVideoOriginalPipeline

from t2v_enhanced.model.diffusers_conditional.models.controlnet.processor import set_use_memory_efficient_attention_xformers
from t2v_enhanced.model.diffusers_conditional.models.controlnet.mask_generator import MaskGenerator

import warnings
# from warnings import warn
from t2v_enhanced.utils.iimage import IImage
from t2v_enhanced.utils.object_loader import instantiate_object
from t2v_enhanced.utils.object_loader import get_class


class VideoLDM(pl.LightningModule):

    def __init__(self,
                 inference_params: pl_module_params_controlnet.InferenceParams,
                 opt_params: pl_module_params_controlnet.OptimizerParams = None,
                 unet_params: pl_module_params_controlnet.UNetParams = None,
                 ):
        super().__init__()

        self.inference_generator = torch.Generator(device=self.device)

        self.opt_params = opt_params
        self.unet_params = unet_params

        print(f"Base pipeline from: {unet_params.pipeline_repo}")
        print(f"Pipeline class {unet_params.pipeline_class}")
        # load entire pipeline (unet, vq, text encoder,..)
        state_dict_control_model = None
        state_dict_fusion = None
        state_dict_base_model = None

        if len(opt_params.load_trained_controlnet_from_ckpt) > 0:
            state_dict_ckpt = torch.load(opt_params.load_trained_controlnet_from_ckpt, map_location=torch.device("cpu"))
            state_dict_ckpt = state_dict_ckpt["state_dict"]
            state_dict_control_model = dict(filter(lambda x: x[0].startswith("unet"), state_dict_ckpt.items()))
            state_dict_control_model = {k.split("unet.")[1]: v for (k, v) in state_dict_control_model.items()}

            state_dict_fusion = dict(filter(lambda x: "cross_attention_merger" in x[0], state_dict_ckpt.items()))
            state_dict_fusion = {k.split("base_model.")[1]: v for (k, v) in state_dict_fusion.items()}
            del state_dict_ckpt

        state_dict_proj = None
        state_dict_ckpt = None

        if hasattr(unet_params, "use_resampler") and unet_params.use_resampler:
            num_queries = unet_params.num_frames if unet_params.num_frames > 1 else None
            if unet_params.use_image_tokens_ctrl:
                num_queries = unet_params.num_control_input_frames
                assert unet_params.frame_expansion == "none"
            image_encoder = self.unet_params.image_encoder
            embedding_dim = image_encoder.embedding_dim

            resampler = instantiate_object(self.unet_params.resampler_cls, video_length=num_queries, embedding_dim=embedding_dim, input_tokens=image_encoder.num_tokens, num_layers=self.unet_params.resampler_merging_layers, aggregation=self.unet_params.aggregation)

            state_dict_proj = None

            self.resampler = resampler
            self.image_encoder = image_encoder

        # print(self.dtype)
        # print(self.unet_params.use_reference_attention)
        # self.dtype = torch.float16
        noise_scheduler = DDPMScheduler.from_pretrained(self.unet_params.pipeline_repo, subfolder="scheduler")
        tokenizer = CLIPTokenizer.from_pretrained(self.unet_params.pipeline_repo, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.unet_params.pipeline_repo, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(self.unet_params.pipeline_repo, subfolder="vae")
        base_model = UNet3DConditionModel.from_pretrained(self.unet_params.pipeline_repo, subfolder="unet", low_cpu_mem_usage=False, device_map=None, merging_mode=self.unet_params.merging_mode_base, use_image_embedding=unet_params.use_resampler and unet_params.use_image_tokens_main, use_fps_conditioning=self.opt_params.use_fps_conditioning, use_reference_attention=self.unet_params.use_reference_attention, unet_params=unet_params)

        if state_dict_base_model is not None:
            miss, unex = base_model.load_state_dict(state_dict_base_model, strict=False)
            assert len(unex) == 0
            if len(miss) > 0:
                warnings.warn(f"Missing keys when loading base_mode:{miss}")
            del state_dict_base_model
        if state_dict_fusion is not None:
            miss, unex = base_model.load_state_dict(state_dict_fusion, strict=False)
            assert len(unex) == 0
            del state_dict_fusion

        print("PIPE LOADING DONE")
        self.noise_scheduler = noise_scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae

        self.unet = ControlNetModel.from_unet(
            unet=base_model,
            conditioning_embedding_out_channels=unet_params.conditioning_embedding_out_channels,
            downsample_controlnet_cond=unet_params.downsample_controlnet_cond,
            num_frames=unet_params.num_frames if (unet_params.frame_expansion != "none" or self.unet_params.use_controlnet_mask) else unet_params.num_control_input_frames,
            num_frame_conditioning=unet_params.num_control_input_frames,
            frame_expansion=unet_params.frame_expansion,
            pre_transformer_in_cond=unet_params.pre_transformer_in_cond,
            num_tranformers=unet_params.num_tranformers,
            vae=AutoencoderKL.from_pretrained(self.unet_params.pipeline_repo, subfolder="vae"),
            zero_conv_mode=unet_params.zero_conv_mode,
            merging_mode=unet_params.merging_mode,
            condition_encoder=unet_params.condition_encoder,
            use_controlnet_mask=unet_params.use_controlnet_mask,
            use_image_embedding=unet_params.use_resampler and unet_params.use_image_tokens_ctrl,
            use_reference_attention=unet_params.use_reference_attention,
            unet_params=unet_params,
            use_image_encoder_normalization=unet_params.use_image_encoder_normalization,
        )
        if state_dict_control_model is not None:
            miss, unex = self.unet.load_state_dict(
                state_dict_control_model, strict=False)
            if len(miss) > 0:
                print("WARNING: Loading checkpoint for controlnet misses states")
                print(miss)

        if unet_params.frame_expansion == "none":
            attention_params = self.unet_params.attention_mask_params
            assert not attention_params.temporal_self_attention_only_on_conditioning and not attention_params.spatial_attend_on_condition_frames and not attention_params.temp_attend_on_neighborhood_of_condition_frames

        self.mask_generator = MaskGenerator(
            self.unet_params.attention_mask_params, num_frame_conditioning=self.unet_params.num_control_input_frames, num_frames=self.unet_params.num_frames)
        self.mask_generator_base = MaskGenerator(
            self.unet_params.attention_mask_params_base, num_frame_conditioning=self.unet_params.num_control_input_frames, num_frames=self.unet_params.num_frames)

        if state_dict_proj is not None and unet_params.use_image_tokens_main:
            if unet_params.use_image_tokens_main:
                missing, unexpected = base_model.load_state_dict(
                    state_dict_proj, strict=False)
            elif unet_params.use_image_tokens_ctrl:
                missing, unexpected = unet.load_state_dict(
                    state_dict_proj, strict=False)
            assert len(unexpected) == 0, f"Unexpected entries {unexpected}"
            print(f"Missing keys state proj = {missing}")
            del state_dict_proj

        base_model.requires_grad_(False)
        self.base_model = base_model
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        layers_config = opt_params.layers_config
        layers_config.set_requires_grad(self)

        print("CUSTOM XFORMERS ATTENTION USED.")
        if is_xformers_available():
            set_use_memory_efficient_attention_xformers(self.unet, num_frame_conditioning=self.unet_params.num_control_input_frames,
                                                        num_frames=self.unet_params.num_frames,
                                                        attention_mask_params=self.unet_params.attention_mask_params
                                                        )
            set_use_memory_efficient_attention_xformers(self.base_model, num_frame_conditioning=self.unet_params.num_control_input_frames,
                                                        num_frames=self.unet_params.num_frames,
                                                        attention_mask_params=self.unet_params.attention_mask_params_base)

        if len(inference_params.scheduler_cls) > 0:
            inf_scheduler_class = get_class(inference_params.scheduler_cls)
        else:
            inf_scheduler_class = DDIMScheduler

        inf_scheduler = inf_scheduler_class.from_pretrained(
            self.unet_params.pipeline_repo, subfolder="scheduler")
        inference_pipeline = TextToVideoSDPipeline(vae=self.vae,
                                                   text_encoder=self.text_encoder,
                                                   tokenizer=self.tokenizer,
                                                   unet=self.base_model,
                                                   controlnet=self.unet,
                                                   scheduler=inf_scheduler
                                                   )

        inference_pipeline.set_noise_generator(self.opt_params.noise_generator)
        inference_pipeline.enable_vae_slicing()

        inference_pipeline.set_progress_bar_config(disable=True)

        self.inference_params = inference_params
        self.inference_pipeline = inference_pipeline

        self.result_processor = ResultProcessor(fps=self.inference_params.frame_rate, n_frames=self.inference_params.video_length)

    def on_start(self):
        datamodule = self.trainer._data_connector._datahook_selector.datamodule
        pipe_id_model = self.unet_params.pipeline_repo
        for dataset_key in ["video_dataset", "image_dataset", "predict_dataset"]:
            dataset = getattr(datamodule, dataset_key, None)
            if dataset is not None and hasattr(dataset, "model_id"):
                pipe_id_data = dataset.model_id
                assert pipe_id_model == pipe_id_data, f"Model and Dataloader need the same pipeline path. Found '{pipe_id_model}' and '{dataset_key}.model_id={pipe_id_data}'. Consider setting '--data.{dataset_key}.model_id={pipe_id_data}'"
        self.result_processor.set_logger(self.logger)

    def on_predict_start(self) -> None:
        self.on_start()
        # pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
        # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        # pipe.set_progress_bar_config(disable=True)
        # self.first_stage = pipe.to(self.device)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        prompts, videos = batch["prompt"], batch["pixel_values"]
        batch_size = videos.shape[0]
        random_p = torch.rand(batch_size)

        videos.to(self.device)
        video_length = videos.shape[1]
        if video_length == 32:
            input_frames_conditioning = videos[:, :self.unet_params.num_frame_conditioning].detach().clone()
            input_frames = videos[:, 16:16+self.unet_params.num_control_input_frames].detach().clone()
            videos = videos[:,16:32].detach().clone()
        else:
            print("Should not be here!")
            input_frames_conditioning = videos[:, :self.unet_params.num_frame_conditioning].detach().clone()
            input_frames = videos[:, :self.unet_params.num_control_input_frames].detach().clone()
        
        pixel_values = rearrange(videos, "B F C W H -> (B F) C W H")
        latents = self.vae.encode(pixel_values).latent_dist.sample() * self.vae.config.scaling_factor
        latents = rearrange(latents, "(B F) C W H -> B C F W H", B=videos.shape[0], F=videos.shape[1])  
        img_cond_resampler = self.resampler if self.unet_params.use_resampler else None
        img_cond_encoder = self.image_encoder if self.unet_params.use_resampler else None

        for i in range(batch_size):
            if random_p[i] < 0.05:
                prompts[i] = ""
            elif random_p[i] > 0.05 and random_p[i] < 0.1:
                input_frames_conditioning[i] = torch.zeros_like(input_frames_conditioning[i])
            elif random_p[i] > 0.1 and random_p[i] < 0.15:
                input_frames_conditioning[i] = torch.zeros_like(input_frames_conditioning[i])
                prompts[i] = ""
            

        if  img_cond_resampler is not None:
            bsz = input_frames.shape[0]
            image_for_conditioniong = rearrange(
                input_frames_conditioning, "B F C W H -> (B F) C W H")
            image_enc = img_cond_encoder(image_for_conditioniong)
            img_cond = img_cond_resampler(image_enc, batch_size=bsz)
            image_enc_unc = img_cond_encoder(
                torch.zeros_like(image_for_conditioniong))
            img_cond_unc = img_cond_resampler(image_enc_unc, batch_size=bsz)
        else:
            img_cond = None
            img_cond_unc = None

        if self.unet_params.use_reference_attention:
            reference_frames_features = self.vae.encode((rearrange(input_frames_conditioning, "B F C W H -> (B F) C W H")).float()).latent_dist.sample() * self.vae.config.scaling_factor
        # reference_frames_features = rearrange(reference_frames_features, "B C W H -> B (W H) C")
            print(f"reference_frames_features: {reference_frames_features.shape}")
        else:
            reference_frames_features = None
 


        prompts_embeds = self.inference_pipeline._encode_prompt(
            prompts, self.device, 1, do_classifier_free_guidance=False, img_cond=img_cond, img_cond_unc=img_cond_unc)
        image, image_vq_enc = self.inference_pipeline.prepare_image(
            image=input_frames, 
            width=self.base_model.config.sample_size * self.vae.config.scaling_factor,
            height=self.base_model.config.sample_size * self.vae.config.scaling_factor,
            batch_size = input_frames.shape[0],
            num_images_per_prompt=1,
            device = self.device,
            dtype = latents.dtype,
            do_classifier_free_guidance=False,
        )
        if len(image.shape) == 5:
            image = rearrange(image, "B F C H W -> (B F) C H W")
        
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, 1000, (bsz,), device=self.device)
        timesteps = timesteps.long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        target = noise 
        with torch.cuda.amp.autocast():
            down_block_res_samples, mid_block_res_sample = self.unet(
                noisy_latents[:, :self.base_model.config.in_channels],
                timesteps,
                encoder_hidden_states=prompts_embeds,
                controlnet_cond=image,
                return_dict=False,
                reference_frames_features=reference_frames_features,
            )
            noise_pred = self.base_model(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompts_embeds,
                down_block_additional_residuals=[
                    sample.to(dtype=latents.dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=latents.dtype),
                fps=None,
                return_dict=False,
                reference_frames_features=reference_frames_features,
            )
            noise_pred = noise_pred[0]
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        # lightning log loss
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        learning_rate = self.opt_params.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        cfg = self.trainer.predict_cfg

        result_file_stem = cfg["result_file_stem"]
        storage_fol = Path(cfg['predict_dir'])
        prompts = cfg["prompt"]
        change_point = cfg["change_point"]
        first_prompt = prompts[0]["prompt"]
        inference_params: pl_module_params_controlnet.InferenceParams = self.inference_params
        conditioning_type = inference_params.conditioning_type
        # n_autoregressive_generations = inference_params.n_autoregressive_generations
        n_autoregressive_generations = cfg["n_autoregressive_generations"]
        mode = inference_params.mode
        start_from_real_input = inference_params.start_from_real_input
        assert isinstance(prompts, list)
        self.inference_generator.manual_seed(self.inference_params.seed)

        assert self.unet_params.num_control_input_frames == self.inference_params.video_length//2, f"currently we assume to have an equal size for and second half of the frame interval, e.g. 16 frames, and we condition on 8. Current setup: {self.unet_params.num_frame_conditioning} and {self.inference_params.video_length}"

        chunks_conditional = []
        batch_size = 1
        shape = (batch_size, self.inference_pipeline.unet.config.in_channels, self.inference_params.video_length,
                 self.inference_pipeline.unet.config.sample_size, self.inference_pipeline.unet.config.sample_size)
        for idx, prompt in enumerate(prompts):
            if idx > 0:
                content = sample*2-1
                content_latent = self.vae.encode(content).latent_dist.sample() * self.vae.config.scaling_factor
                content_latent = rearrange(content_latent, "F C W H -> 1 C F W H")
                content_latent = content_latent[:, :, self.unet_params.num_control_input_frames:].detach().clone()

            if hasattr(self.inference_pipeline, "noise_generator"):
                latents = self.inference_pipeline.noise_generator.sample_noise(shape=shape, device=self.device, dtype=self.dtype, generator=self.inference_generator, content=content_latent if idx > 0 else None)
            else:
                latents = None
            if idx == 0:
                # sample = self.generate_first_sample(prompt)
                # print(sample.shape)
                # print(sample)
                sample = cfg["video"].to(self.device)
            else:
                if inference_params.conditioning_type == "fixed":
                    context = chunks_conditional[0][:self.unet_params.num_frame_conditioning]
                    context = [context]
                    context = [2*sample-1 for sample in context]

                    input_frames_conditioning = torch.cat(context).detach().clone()
                    input_frames_conditioning = rearrange(input_frames_conditioning, "F C W H -> 1 F C W H")
                    # print(f"input_frames_conditioning shape: {input_frames_conditioning.shape}")
                elif inference_params.conditioning_type == "last_chunk":
                    input_frames_conditioning = condition_input[:, -self.unet_params.num_frame_conditioning:].detach().clone()
                elif inference_params.conditioning_type == "past":
                    context = [sample[:self.unet_params.num_control_input_frames] for sample in chunks_conditional]
                    context = [2*sample-1 for sample in context]

                    input_frames_conditioning = torch.cat(context).detach().clone()
                    input_frames_conditioning = rearrange(input_frames_conditioning, "F C W H -> 1 F C W H")
                else:
                    raise NotImplementedError()

                input_frames = condition_input[:, self.unet_params.num_control_input_frames:].detach().clone()
                if idx in change_point or  idx - 1 in change_point or idx - 2 in change_point:
                    controlnet_weight = 0.8
                else:
                    controlnet_weight = 1
                sample = self(prompt, input_frames=input_frames, input_frames_conditioning=input_frames_conditioning, latents=latents, controlnet_weight=controlnet_weight)
                # print(f"sample shape: {sample.shape}") 16*3*256*256
            if hasattr(self.inference_pipeline, "reset_noise_generator_state"):
                self.inference_pipeline.reset_noise_generator_state()

            condition_input = rearrange(sample, "F C W H -> 1 F C W H")
            condition_input = (2*condition_input)-1  # range: [-1,1]

            # store first 16 frames, then always last 8 of a chunk
            chunks_conditional.append(sample)

        result_formats = self.inference_params.result_formats
        # result_formats = [gif", "mp4"]
        concat_video = self.inference_params.concat_video

        def IImage_normalized(x): return IImage(x, vmin=0, vmax=1)
        for result_format in result_formats:
            save_format = result_format.replace("eval_", "")

            merged_video = None
            for chunk_idx, (prompt, video) in enumerate(zip(prompts, chunks_conditional)):
                if chunk_idx == 0:
                    current_video = IImage_normalized(video)
                else:
                    current_video = IImage_normalized(video[self.unet_params.num_control_input_frames:])

                if merged_video is None:
                    merged_video = current_video
                else:
                    merged_video &= current_video

            if concat_video:
                filename = video_naming(first_prompt, save_format, batch_idx, 0)
                result_file_video = (storage_fol / filename).absolute().as_posix()
                result_file_video = (Path(result_file_video).parent / (result_file_stem+Path(result_file_video).suffix)).as_posix()
                self.result_processor.save_to_file(video=merged_video.torch(vmin=0, vmax=1), prompt=first_prompt, video_filename=result_file_video, prompt_on_vid=False)

    def forward(self, prompt, input_frames=None, input_frames_conditioning=None, latents=None, controlnet_weight=1.0):
        call_params = self.inference_params.to_dict()
        # print(f"INFERENCE PARAMS = {call_params}")
        call_params["prompt"] = prompt
        call_params["controlnet_weight"] = controlnet_weight
        call_params["negative_prompt"] = "worst quality, low quality, medium quality, deleted, lowres, comic, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
        call_params["image"] = input_frames
        call_params["num_frames"] = self.inference_params.video_length
        call_params["return_dict"] = False
        call_params["output_type"] = "pt_t2v"
        call_params["mask_generator"] = self.mask_generator
        call_params["precision"] = "16" if self.trainer.precision.startswith("16") else "32"
        call_params["no_text_condition_control"] = self.opt_params.no_text_condition_control
        call_params["weight_control_sample"] = self.unet_params.weight_control_sample
        call_params["use_controlnet_mask"] = self.unet_params.use_controlnet_mask
        call_params["skip_controlnet_branch"] = self.opt_params.skip_controlnet_branch
        call_params["img_cond_resampler"] = self.resampler if self.unet_params.use_resampler else None
        call_params["img_cond_encoder"] = self.image_encoder if self.unet_params.use_resampler else None
        call_params["input_frames_conditioning"] = input_frames_conditioning
        call_params["cfg_text_image"] = self.unet_params.cfg_text_image
        call_params["use_of"] = self.unet_params.use_of
        call_params["use_reference_attention"] = self.unet_params.use_reference_attention
        # print(call_params["use_reference_attention"])
        if latents is not None:
            call_params["latents"] = latents

        sample = self.inference_pipeline(generator=self.inference_generator, **call_params)
        return sample
    
    def generate_first_sample(self, prompt):

        original_pipeline = TextToVideoOriginalPipeline(vae=self.vae,
                                                   text_encoder=self.text_encoder,
                                                   tokenizer=self.tokenizer,
                                                   unet=self.base_model,
                                                   scheduler=self.noise_scheduler
                                                   )

        # original_pipeline.set_noise_generator(self.opt_params.noise_generator)
        original_pipeline.enable_vae_slicing()

        original_pipeline.set_progress_bar_config(disable=True)
        # call_params = self.inference_params.to_dict()
        frames = original_pipeline(prompt=prompt, generator=self.inference_generator, num_inference_steps=50,
                                   eta=1.0, height=256, width=256).frames
        frames = torch.stack([torch.from_numpy(frame) for frame in frames])
        frames = frames.to(self.device).to(torch.float32)
        del original_pipeline
        return rearrange(frames[0], "F W H C -> F C W H")

        # del original_pipeline
        # return sample
