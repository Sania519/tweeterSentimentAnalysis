from flask import Flask, request, jsonify, send_file
import os
import json
import uuid
from datetime import datetime
import pickle
import io
from google.cloud import storage
import numpy as np
import pandas as pd
from textblob import TextBlob
import re

app = Flask(__name__)

# Model storage dictionary to keep track of models in memory
model_registry = {}

# Configure Google Cloud Storage
# Make sure to set GOOGLE_APPLICATION_CREDENTIALS environment variable 
# or use service account key file
storage_client = storage.Client()
BUCKET_NAME = "twitter-sentiment-models"  # Replace with your actual bucket name

# Create bucket if it doesn't exist (with private access)
def ensure_bucket_exists():
    try:
        bucket = storage_client.get_bucket(BUCKET_NAME)
    except Exception:
        # Create a new bucket with private access (no public access)
        bucket = storage_client.create_bucket(BUCKET_NAME)
        # Ensure the bucket is private
        bucket.iam_configuration.public_access_prevention = "enforced"
        bucket.patch()
    return bucket

# Text cleaning function for tweets
def clean_tweet(tweet):
    """
    Clean tweet text by removing links, special characters
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

# Simple sentiment analysis using TextBlob
def analyze_sentiment(tweet_text):
    """
    Analyze the sentiment of a tweet using TextBlob
    Returns: polarity (-1 to 1) and subjectivity (0 to 1)
    """
    cleaned_tweet = clean_tweet(tweet_text)
    analysis = TextBlob(cleaned_tweet)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Classify sentiment based on polarity
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment
    }

@app.route('/model', methods=['POST'])
def upload_model():
    """
    API Call 1: Upload a model
    Input: Model data
    Output: OK response with model identifier
    """
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400
    
    model_file = request.files['model']
    
    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"model_{timestamp}_{uuid.uuid4()}"
    filename = f"{model_id}.pkl"
    
    # Ensure the bucket exists
    bucket = ensure_bucket_exists()
    
    # Upload to GCP bucket
    blob = bucket.blob(filename)
    blob.upload_from_file(model_file)
    
    # Load model into memory
    model_file.seek(0)  # Reset file pointer to beginning
    model = pickle.load(model_file)
    
    # Store reference in memory
    model_registry[model_id] = {
        "model": model,
        "filename": filename,
        "uploaded_at": timestamp
    }
    
    # Print the model ID for reference
    print(f"Model uploaded successfully. Model ID: {model_id}")
    
    return jsonify({
        "status": "OK",
        "message": "Model uploaded successfully",
        "model_id": model_id
    })

@app.route('/analyze', methods=['POST'])
def analyze_tweet():
    """
    API Call 2: Analyze tweet sentiment
    Input: JSON with tweet text
    Output: Sentiment analysis results
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        if 'tweet' not in data:
            return jsonify({"error": "No tweet text provided"}), 400
        
        tweet_text = data['tweet']
        
        # If model_id is provided, use the model for prediction
        if 'model_id' in data and data['model_id']:
            model_id = data['model_id']
            
            # Check if model is in memory
            if model_id not in model_registry:
                return jsonify({"error": f"Model {model_id} not found"}), 404
            
            # Use the model to make prediction
            model = model_registry[model_id]["model"]
            
            # Process with the model (depends on model type)
            try:
                # Assuming model predicts sentiment class
                prediction = model.predict([tweet_text])[0]
                
                # Map model prediction to sentiment
                if isinstance(prediction, (int, np.integer)):
                    if prediction == 0:
                        sentiment = "negative"
                    elif prediction == 1:
                        sentiment = "neutral"
                    else:
                        sentiment = "positive"
                else:
                    sentiment = str(prediction)
                
                return jsonify({
                    "tweet": tweet_text,
                    "sentiment": sentiment,
                    "using_model": True,
                    "model_id": model_id
                })
            
            except Exception as e:
                # Fallback to TextBlob if model prediction fails
                result = analyze_sentiment(tweet_text)
                result["tweet"] = tweet_text
                result["using_model"] = False
                result["error"] = str(e)
                return jsonify(result)
        
        # No model provided, use TextBlob
        result = analyze_sentiment(tweet_text)
        result["tweet"] = tweet_text
        result["using_model"] = False
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/model/<model_id>', methods=['GET'])
def get_model(model_id):
    """
    API Call 3: Get model
    Fetches model from GCP storage bucket and returns it
    """
    try:
        # Check if model_id exists
        if model_id not in model_registry:
            return jsonify({"error": f"Model {model_id} not found"}), 404
        
        # Get model metadata
        model_metadata = model_registry[model_id]
        filename = model_metadata["filename"]
        
        # Get from GCP bucket
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        
        # Download to memory
        model_bytes = io.BytesIO()
        blob.download_to_file(model_bytes)
        model_bytes.seek(0)  # Reset to beginning of file
        
        # Return the model file
        return send_file(
            model_bytes,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a route for batch analysis of tweets
@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Batch analyze multiple tweets
    Input: JSON with array of tweets
    Output: Array of sentiment analysis results
    """
    try:
        data = request.json
        
        if not data or 'tweets' not in data:
            return jsonify({"error": "No tweets provided"}), 400
        
        tweets = data['tweets']
        model_id = data.get('model_id', None)
        
        results = []
        
        for tweet in tweets:
            if model_id and model_id in model_registry:
                # Use model for prediction
                model = model_registry[model_id]["model"]
                try:
                    prediction = model.predict([tweet])[0]
                    
                    # Map model prediction to sentiment
                    if isinstance(prediction, (int, np.integer)):
                        if prediction == 0:
                            sentiment = "negative"
                        elif prediction == 1:
                            sentiment = "neutral"
                        else:
                            sentiment = "positive"
                    else:
                        sentiment = str(prediction)
                    
                    results.append({
                        "tweet": tweet,
                        "sentiment": sentiment,
                        "using_model": True
                    })
                except Exception:
                    # Fallback to TextBlob
                    result = analyze_sentiment(tweet)
                    result["tweet"] = tweet
                    result["using_model"] = False
                    results.append(result)
            else:
                # Use TextBlob
                result = analyze_sentiment(tweet)
                result["tweet"] = tweet
                result["using_model"] = False
                results.append(result)
        
        return jsonify({
            "results": results,
            "count": len(results),
            "using_model": model_id is not None and model_id in model_registry,
            "model_id": model_id if model_id in model_registry else None
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add a route to list all available models
@app.route('/models', methods=['GET'])
def list_models():
    """
    List all models currently in the registry
    Output: List of model IDs and their upload timestamps
    """
    models_list = []
    for model_id, model_data in model_registry.items():
        models_list.append({
            "model_id": model_id,
            "uploaded_at": model_data["uploaded_at"],
            "filename": model_data["filename"]
        })
    
    return jsonify({
        "models": models_list,
        "count": len(models_list)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))