📊 Athlete Analysis App
This project uses AI to analyze athlete performance from video footage, focusing on posture, form, and predictive movement insights.

🧠 Overview
The repository consists of:
* AItest.py: A Python script that:
Extracts frames from a video at regular intervals.
Encodes them in Base64 format.
Sends them in batches to OpenAI's GPT-4o model for frame-by-frame athletic performance evaluation and prediction.
* AthleteAnalysisApp/: (Folder contents not yet detailed; assumed to be related to a front-end or GUI interface for displaying analysis results.)

📂 File Structure
bash
Copy
Edit
.
├── AItest.py                # Main script for video processing and AI analysis
├── AthleteAnalysisApp/     # (Presumed front-end or support files)
└── sampled_api_responses.json (Generated)  # Output file with AI feedback

🧪 Requirements
Python 3.8+
OpenAI SDK (openai)
OpenCV (cv2)
Async libraries: aiohttp, asyncio
numpy, base64, tqdm

Install all with:
bash
Copy
Edit
pip install openai opencv-python aiohttp tqdm numpy

🎥 Usage
Replace the video file path in AItest.py (VIDEO_PATH = "boxt.mp4") with your desired file.
Set your OpenAI API key in API_KEY.

Run the script:
bash
Copy
Edit
python AItest.py
Review the output in sampled_api_responses.json.

⚙️ Settings
You can configure:
SAMPLE_RATE: How frequently to extract frames.
BATCH_SIZE: Number of frames sent per API call.
MAX_CONCURRENT_REQUESTS: API call concurrency.
