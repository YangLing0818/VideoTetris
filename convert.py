from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(mp4_path, gif_path):
    clip = VideoFileClip(mp4_path)
    clip.write_gif(gif_path)

# convert all files in ./assests
import os
for file in os.listdir('./assets'):
    if file.endswith('.mp4'):
        mp4_path = os.path.join('./assets', file)
        gif_path = os.path.join('./assets', file.replace('.mp4', '.gif'))
        convert_mp4_to_gif(mp4_path, gif_path)
        