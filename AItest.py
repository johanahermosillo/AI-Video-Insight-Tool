import json
import asyncio
import aiohttp
import cv2
import base64
import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm
import time
import os
from concurrent.futures import ThreadPoolExecutor

# Configuration settings
API_KEY = "YOUR API KEY    "
VIDEO_PATH = "boxt.mp4"
SAMPLE_RATE = 20
BATCH_SIZE = 5
MAX_CONCURRENT_REQUESTS = 3
OUTPUT_FILENAME = "sampled_api_responses.json"
DISPLAY_FRAMES = False  # Set to True if you want to see frames
MODEL = "gpt-4o"
MAX_TOKENS = 500

client = AsyncOpenAI(api_key=API_KEY)


async def extract_frames():
    """Extract frames from video with optimized processing"""
    print("Starting frame extraction...")
    base64_frames = []

    # Get video properties
    video = cv2.VideoCapture(VIDEO_PATH)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {VIDEO_PATH}")

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_frames = total_frames // SAMPLE_RATE

    # Use ThreadPoolExecutor for image processing (CPU-bound tasks)
    with ThreadPoolExecutor() as executor:
        frame_index = 0
        with tqdm(total=sampled_frames, desc="Extracting frames") as pbar:
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break

                if frame_index % SAMPLE_RATE == 0:
                    # Process frame in thread pool
                    def process_frame(frame):
                        _, buffer = cv2.imencode(".jpg", frame)
                        return base64.b64encode(buffer)

                    base64_frames.append(await asyncio.get_event_loop().run_in_executor(
                        executor, process_frame, frame))

                    if DISPLAY_FRAMES:
                        cv2.imshow('Frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    pbar.update(1)

                frame_index += 1

    video.release()
    if DISPLAY_FRAMES:
        cv2.destroyAllWindows()

    print(f"{len(base64_frames)} sampled frames extracted.")
    return base64_frames


async def process_batch(batch_number, frame_indexes, batch_frames):
    """Process a single batch of frames with the API"""
    try:
        message_content = [{"type": "text", "text": "Describe what is happening in these frames and give advice based on form and posture along with trying to predict what might happen next:"}]

        for frame_bytes in batch_frames:
            base64_string = frame_bytes.decode('utf-8')
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_string}"
                }
            })

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=MAX_TOKENS
        )

        result = {
            "batch_number": batch_number,
            "frame_indexes": frame_indexes,
            "response": response.choices[0].message.content
        }

        # Print the response to console immediately
        print(f"\n--- Batch {batch_number} (Frames {frame_indexes[0]}-{frame_indexes[-1]}) ---")
        print(result["response"])
        print("-" * 50)

        return result

    except Exception as e:
        print(f"Error processing batch {batch_number}: {str(e)}")
        # Add exponential backoff retry logic here if needed
        return {
            "batch_number": batch_number,
            "frame_indexes": frame_indexes,
            "response": f"Error: {str(e)}"
        }


async def print_all_responses(responses):
    """Print all responses in order to console"""
    print("\n\n========== ALL RESPONSES ==========")
    for resp in sorted(responses, key=lambda x: x["batch_number"]):
        print(f"\nBatch {resp['batch_number']} (Frames {resp['frame_indexes'][0]}-{resp['frame_indexes'][-1]}):")
        print(resp["response"])
        print("-" * 50)


async def main():
    start_time = time.time()

    # Extract all frames first
    base64_frames = await extract_frames()

    # Prepare batches
    batches = []
    for i in range(0, len(base64_frames), BATCH_SIZE):
        batch_frames = base64_frames[i:i + BATCH_SIZE]
        frame_indexes = list(range(i, min(i + BATCH_SIZE, len(base64_frames))))
        batches.append((i // BATCH_SIZE, frame_indexes, batch_frames))

    # Process batches with controlled concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process_with_semaphore(batch_data):
        async with semaphore:
            return await process_batch(*batch_data)

    print(f"Processing {len(batches)} batches with max {MAX_CONCURRENT_REQUESTS} concurrent requests...")
    tasks = [process_with_semaphore(batch_data) for batch_data in batches]

    responses = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batches"):
        result = await task
        responses.append(result)

    # Sort responses by batch number to maintain order
    responses.sort(key=lambda x: x["batch_number"])

    # Print all responses in sequence again at the end (optional)
    await print_all_responses(responses)

    # Save responses to file
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print(f"Saved {len(responses)} batch responses to {OUTPUT_FILENAME}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
