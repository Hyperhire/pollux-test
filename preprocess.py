from utils.pollux_utils import frame_video,segment_frames_from_video

if __name__ == "__main__":
    import sys
    video_name = sys.argv[1]
    frame_video(video_name)
    segment_frames_from_video(video_name)