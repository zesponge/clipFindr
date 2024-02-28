from google.cloud import videointelligence
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

input_uri = "gs://scrobblesearch/subAd.mp4"

@app.get('/gptcall')
def gptcall():
    gptRes = gpt_check()
    return(gptRes)

@app.get('/transcribe')
def transcribeCall():
    transcript = transcribe_video(input_uri)
    return transcript

@app.get('/annotate')
def annotateCall():
    annotation = annotate_video(input_uri)
    return ("annotation done")


def transcribe_video(input_uri):
    # set key credentials file path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\sumar\Desktop\opportune-ego-415620-d9343252e488.json"

    """Transcribe speech from a video stored on GCS bucket."""

    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]

    config = videointelligence.SpeechTranscriptionConfig(
        language_code="en-US", enable_automatic_punctuation=True
    )
    video_context = videointelligence.VideoContext(speech_transcription_config=config)

    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_uri": input_uri,
            "video_context": video_context,
        }
    )

    print("\nProcessing video for speech transcription.")

    result = operation.result(timeout=600)

    # There is only one annotation_result since only
    # one video is processed.
    annotation_results = result.annotation_results[0]
    for speech_transcription in annotation_results.speech_transcriptions:

        # The number of alternatives for each transcription is limited by
        # SpeechTranscriptionConfig.max_alternatives.
        # Each alternative is a different possible transcription
        # and has its own confidence score.
        for alternative in speech_transcription.alternatives:
            print("full transcript:")

            print("Transcript: {}".format(alternative.transcript))
            # print("Confidence: {}\n".format(alternative.confidence))

            # print("everything:")
            # print(alternative)
            
            print("group information:")
            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time
                end_time = word_info.end_time
                print(
                    "\t{}s - {}s: {}".format(
                        start_time.seconds + start_time.microseconds * 1e-6,
                        end_time.seconds + end_time.microseconds * 1e-6,
                        word,
                    )
                )
    return alternative.transcript


def annotate_video(input_uri):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\sumar\Desktop\opportune-ego-415620-d9343252e488.json"

    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.LABEL_DETECTION]

    mode = videointelligence.LabelDetectionMode.SHOT_AND_FRAME_MODE
    config = videointelligence.LabelDetectionConfig(label_detection_mode=mode)
    context = videointelligence.VideoContext(label_detection_config=config)

    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_uri": input_uri,
            "video_context": context,
        }
    )

    print("\nProcessing video for label annotations:")

    result = operation.result(timeout=180)
    print("\nlabel annotations Finished processing.\n")

    # Process shot level label annotations
    shot_labels = result.annotation_results[0].shot_label_annotations
    for i, shot_label in enumerate(shot_labels):
        print("Shot label description: {}".format(shot_label.entity.description))
        for category_entity in shot_label.category_entities:
            print(
                # "\tLabel category description: {}".format(category_entity.description)
            )

        for i, shot in enumerate(shot_label.segments):
            start_time = (
                shot.segment.start_time_offset.seconds
                + shot.segment.start_time_offset.microseconds / 1e6
            )
            end_time = (
                shot.segment.end_time_offset.seconds
                + shot.segment.end_time_offset.microseconds / 1e6
            )
            positions = "{}s to {}s".format(start_time, end_time)
            # confidence = shot.confidence
            print("\tSegment {}: {}".format(i, positions))
            # print("\tConfidence: {}".format(confidence))
        print("\n")

def gpt_check():

    data_string = """
Shot label description: glasses
        Segment 0: 9.217541s to 10.26025s  
        Segment 1: 12.804458s to 13.888875s


Shot label description: cuisine

        Segment 0: 13.930583s to 15.557208s
        Segment 1: 15.598916s to 16.891875s


Shot label description: cake

        Segment 0: 10.301958s to 11.219541s


Shot label description: conversation       

        Segment 0: 9.217541s to 10.26025s  
        Segment 1: 18.059708s to 22.77275s 
        Segment 2: 23.982291s to 25.442083s


Shot label description: fast food

        Segment 0: 16.933583s to 18.018s
        Segment 1: 26.77675s to 28.194833s
        Segment 2: 28.236541s to 29.988291s


Shot label description: cheeseburger

        Segment 0: 16.933583s to 18.018s


Shot label description: hamburger

        Segment 0: 16.933583s to 18.018s
        Segment 1: 26.77675s to 28.194833s
        Segment 2: 28.236541s to 29.988291s


Shot label description: senior citizen

        Segment 0: 9.217541s to 10.26025s
        Segment 1: 18.059708s to 22.77275s
        Segment 2: 23.982291s to 25.442083s


Shot label description: facial expression

        Segment 0: 23.982291s to 25.442083s


Shot label description: junk food

        Segment 0: 16.933583s to 18.018s
        Segment 1: 28.236541s to 29.988291s


Shot label description: meat

        Segment 0: 3.5035s to 5.25525s


Shot label description: eating

        Segment 0: 13.930583s to 15.557208s
        Segment 1: 16.933583s to 18.018s


Shot label description: cooking

        Segment 0: 3.5035s to 5.25525s
        Segment 1: 13.930583s to 15.557208s


Shot label description: sandwich

        Segment 0: 16.933583s to 18.018s
        Segment 1: 26.77675s to 28.194833s
        Segment 2: 28.236541s to 29.988291s


Shot label description: finger food

        Segment 0: 26.77675s to 28.194833s


Shot label description: television advertisement
        Segment 0: 12.804458s to 13.888875s


Shot label description: food
        Segment 0: 3.5035s to 5.25525s
        Segment 1: 10.301958s to 11.219541s
        Segment 2: 11.26125s to 12.76275s
        Segment 3: 12.804458s to 13.888875s
        Segment 4: 13.930583s to 15.557208s
        Segment 5: 15.598916s to 16.891875s
        Segment 6: 16.933583s to 18.018s
        Segment 7: 26.77675s to 28.194833s
        Segment 8: 28.236541s to 29.988291s
    """

    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a video analysis tool used to tell users during what time segments specific objects appear."},
        {"role": "user", "content": "given the following text: " + data_string + " what are the important objects in the video?"},
    ]
    )

    print(completion.choices[0].message.content)
    return(completion.choices[0].message.content)
    
    

if __name__ == "__main__":
    app.run()

    # Transcribe the video and store the result in a variable
    # video_transcription = transcribe_video(input_uri)
    # annotate_video(input_uri)
    # gpt_check()