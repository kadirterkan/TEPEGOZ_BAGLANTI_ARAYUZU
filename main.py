import argparse
import concurrent.futures
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from decouple import config

from src.connection_handler import ConnectionHandler
from src.frame_predictions import FramePredictions
from src.object_detection_model import ObjectDetectionModel


def configure_logger(team_name):
    log_folder = "./_logs/"
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def run():
    print("Started...")
    # Get configurations from .env file
    config.search_path = "./config/"
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")

    # Declare logging configuration.
    configure_logger(team_name)

    # Teams can implement their codes within ObjectDetectionModel class. (OPTIONAL)
    detection_model = ObjectDetectionModel(evaluation_server_url)

    # Connect to the evaluation server.
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)

    # Get all frames from current active session.
    frames_json = server.get_frames()

    # Create images folder
    images_folder = "./_images/"
    Path(images_folder).mkdir(parents=True, exist_ok=True)

    # Run object detection model frame by frame.
    for frame in frames_json:

        if os.path.isfile(server.sent_folder + server.filename):
            with open(server.sent_folder + server.filename, 'r') as f:
                data = f.read().splitlines()
                if str(frame['image_url'].split("/")[-1]) + "@" + str(frame['video_name']) in data:
                    print("SKIPPING" + frame['image_url'].split("/")[-1])
                    continue

        # Create a prediction object to store frame info and detections
        predictions = FramePredictions(frame['url'], frame['image_url'], frame['video_name'])

        # Run detection model
        predictions = detection_model.process(predictions)
        
        mode = 'a' if os.path.exists(server.sent_folder + server.filename) else 'w'
        with open(server.sent_folder + server.filename, mode) as f:
            f.write(predictions.image_url.split("/")[-1] + "@" + predictions.video_name + '\n')

        # Send model predictions of this frame to the evaluation server
        if len(predictions.detected_objects) != 0:
            result = server.send_prediction(predictions)

            if result.status_code != 201:
                response_json = json.loads(result.text)
                while "You do not have permission to perform this action." in response_json:
                    result = server.send_prediction(predictions)
                    response_json = json.loads(result.text)

def test():
    config.search_path = "./config/"

    evaluation_server_url = config("EVALUATION_SERVER_URL")

    detection_model = ObjectDetectionModel(evaluation_server_url)

    for path in Path("./_images/").iterdir():
        if path.is_file():
            detection_model.test_detect(path.name)
            print(path.name)


if __name__ == '__main__':
    run()
    # test()
