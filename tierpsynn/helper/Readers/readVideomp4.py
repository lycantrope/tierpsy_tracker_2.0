import cv2


class readVideomp4:
    def __init__(self, video_file):
        # Open the video file using OpenCV
        self.vid = cv2.VideoCapture(video_file)

        if not self.vid.isOpened():
            raise ValueError(f"Error opening video file: {video_file}")

        # Get properties of the video
        self.frame_max = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        self.first_frame = 0
        self.tot_frames = self.frame_max - self.first_frame + 1

        # Get frame width, height, and type
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)
        _, img = self.vid.read()  # Read first frame to get dtype
        self.dtype = img.dtype if img is not None else None

        # Reset video to first frame
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frames_read = []

    def read(self):
        # Read the next frame in the video
        ret, img = self.vid.read()
        if not ret:
            return False, None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_number = int(self.vid.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame_timestamp = frame_number / self.fps  # Approximate timestamp
        self.frames_read.append((frame_number, frame_timestamp))
        return True, img

    def read_frame(self, frame_number):
        # Jump to the specific frame number and read it
        if frame_number >= self.frame_max:
            return False, None

        self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, img = self.vid.read()
        if not ret:
            return False, None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_timestamp = frame_number / self.fps
        self.frames_read.append((frame_number, frame_timestamp))
        return True, img

    def __len__(self):
        # Return the total number of frames in the video
        return self.frame_max - self.first_frame + 1

    def release(self):
        # Release the video file
        return self.vid.release()
