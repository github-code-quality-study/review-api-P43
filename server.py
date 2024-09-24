import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

# The server should only return results filtered by location for these locations:
LOCATIONS = ["Albuquerque, New Mexico",
"Carlsbad, California",
"Chula Vista, California",
"Colorado Springs, Colorado",
"Denver, Colorado",
"El Cajon, California",
"El Paso, Texas",
"Escondido, California",
"Fresno, California",
"La Mesa, California",
"Las Vegas, Nevada",
"Los Angeles, California",
"Oceanside, California",
"Phoenix, Arizona",
"Sacramento, California",
"Salt Lake City, Utah",
"Salt Lake City, Utah",
"San Diego, California",
"Tucson, Arizona"]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(reviews, indent=2).encode("utf-8")
            
            # The reviews will need to be analyzed for sentiment using the method analyze_sentiment; this method will not need to be updated or edited in any way and can be assumed to work as intended. The analyze_sentiment method returns a dictionary with four items: 'neg', 'neu', 'pos', and 'compound'. These represent the negative, neutral, positive, and compound sentiment scores of the input text, respectively.
            for review in reviews:
                review_body = review["ReviewBody"]
                sentiment_scores = self.analyze_sentiment(review_body)
                review["sentiment"] = sentiment_scores

            # get the query string from the environ dictionary
            query_string = environ["QUERY_STRING"]
            location = parse_qs(query_string).get("location", None)
            start_date = parse_qs(query_string).get("start_date", None)
            end_date = parse_qs(query_string).get("end_date", None)

            filtered_reviews = reviews

            #  filter by location
            if location:
                filtered_reviews = [review for review in filtered_reviews if review["Location"] == location and location in LOCATIONS]

            # To view the reviews for a specific date range you can include the start_date and end_date parameters in the URL.
            # filter by start_date
            if start_date:
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], TIMESTAMP_FORMAT) >= datetime.strptime(start_date[0], '%Y-%m-%d')]

            # filter by end_date
            if end_date:
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], TIMESTAMP_FORMAT) <= datetime.strptime(end_date[0], '%Y-%m-%d')]
                
            # Sort by sentiment compound score, descending
            filtered_reviews = sorted(filtered_reviews, key=lambda x: x["sentiment"]["compound"], reverse=True)

            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            
            # Parse the request body
            request_body_size = int(environ.get("CONTENT_LENGTH", 0))
            request_body = environ["wsgi.input"].read(request_body_size).decode("utf-8")
            parsed_body = parse_qs(request_body)

            location = parsed_body.get("Location", [None])[0]
            review_body = parsed_body.get("ReviewBody", [None])[0]

            if location is None:
                start_response("400 Bad Request", [])
                return [b'missing location']
            
            if review_body is None:
                start_response("400 Bad Request", [])
                return [b'missing review body']

            if location not in LOCATIONS:
                start_response("400 Bad Request", [])
                return [b'invalid location']
            
            # Create a new review dictionary
            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": datetime.now().strftime(TIMESTAMP_FORMAT)
            }

            # Append the new review to the reviews list
            reviews.append(new_review)

            # Create the response body from the new review and convert to a JSON byte string
            response_body = json.dumps(new_review, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()