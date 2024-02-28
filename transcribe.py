from google.cloud import videointelligence

def transcribe_video(input_url):
   """Transcribe a video from a Google Cloud Storage bucket."""
   client = videointelligence.VideoIntelligenceServiceClient()

    # Define features to be extracted
   features = [videointelligence.enums.Feature.TEXT_DETECTION]

    # Configure the request
   config = videointelligence.types.TextDetectionConfig()
   config.language_hints.append("en-US")

    # Construct the video context
   context = videointelligence.types.VideoContext(text_detection_config=config)

    # Perform the request
   operation = client.annotate_video(input_uri=input_url, features=features, video_context=context)

    # Wait for the operation to complete
   result = operation.result(timeout=900)

    # Extract the transcription results
   transcription = []
   for annotation_result in result.annotation_results:
    for text_annotation in annotation_result.text_annotations:
       transcription.append(text_annotation.text)

    return transcription

if __name__ == "__main__":
    input_url = "gs://scrobblesearch/doordashAd.mp4"
    video_transcription = transcribe_video(input_url)
    for text in video_transcription:
        print(text)