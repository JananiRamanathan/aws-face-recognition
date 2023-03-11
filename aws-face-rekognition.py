import io
import boto3
from PIL import Image, ImageDraw, ImageFont

def analyze_local_image(rek_client, model, photo, min_confidence):
    image = Image.open(photo)

    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes = image_bytes.getvalue()

    response = rek_client.detect_custom_labels(Image={'Bytes': image_bytes},
                                                   MinConfidence=min_confidence,
                                                   ProjectVersionArn=model)
    return response['CustomLabels']

def main():
    min_confidence = 50

    session = boto3.Session(profile_name='default')
    rekognition_client = session.client("rekognition")

    label = analyze_local_image(rekognition_client,
                                              "arn:aws:rekognition:us-east-1:160071257600:project/face-poc/version/face-poc.2023-03-11T13.15.13/1678520713612",
                                              "faces/face2.jpg",
                                              min_confidence)
    print(label)
if __name__ == "__main__":
    main()
