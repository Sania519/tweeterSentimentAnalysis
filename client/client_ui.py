from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
import requests
import os
import pickle
import json
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime

# API base URL - Get from environment variable or use default
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:5001')  # Make sure this matches your backend port

# Upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Results folder for analysis reports
RESULTS_FOLDER = 'results'
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# Create the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.secret_key = 'twitter_sentiment_secret_key'  # Needed for flashing messages

@app.route('/')
def index():
    """Home page with options to analyze tweets, upload models, and view results."""
    # Get tab parameter, default to 'single_tweet' if current_model_id exists, otherwise 'model_management'
    tab = request.args.get('tab')
    if not tab:
        tab = 'single_tweet' if session.get('current_model_id') else 'model_management'
    
    return render_template('index.html', 
                          model_id=session.get('current_model_id'), 
                          api_url=API_BASE_URL,
                          active_tab=tab)

@app.route('/upload', methods=['POST'])
def upload_model():
    """Upload a model to the API service."""
    
    if 'model_file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index', tab='model_management'))
    
    model_file = request.files['model_file']
    
    if model_file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index', tab='model_management'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(model_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        model_file.save(file_path)
        
        # Upload to API - use multipart/form-data
        files = {'model': open(file_path, 'rb')}
        response = requests.post(f"{API_BASE_URL}/model", files=files)
        
        if response.status_code == 200:
            result = response.json()
            # Store model_id in session
            session['current_model_id'] = result['model_id']
            flash(f'Model uploaded successfully. Model ID: {result["model_id"]}', 'success')
            
            # Close the file after use
            files['model'].close()
            
            # Redirect to the Single Tweet tab
            return redirect(url_for('index', tab='single_tweet'))
        else:
            flash(f'Error uploading model: {response.text}', 'error')
            files['model'].close()
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index', tab='model_management'))
    
    return redirect(url_for('index', tab='model_management'))

@app.route('/analyze', methods=['POST'])
def analyze_tweet():
    """Analyze a single tweet's sentiment."""
    
    tweet_text = request.form.get('tweet_text', '')
    
    if not tweet_text:
        flash('No tweet text provided', 'error')
        return redirect(url_for('index', tab='single_tweet'))
    
    try:
        # Format data to match the API expectation - 'tweet' instead of 'sample'
        data = {
            'tweet': tweet_text
        }
        
        # Add model_id if available
        if session.get('current_model_id'):
            data['model_id'] = session.get('current_model_id')
        
        # Call the /analyze endpoint (not /predict)
        response = requests.post(f"{API_BASE_URL}/analyze", json=data)
        
        if response.status_code == 200:
            result = response.json()
            # Get sentiment directly from the response
            sentiment = result.get('sentiment', 'unknown')
            polarity = result.get('polarity', 0)
            subjectivity = result.get('subjectivity', 0)
            
            sentiment_class = ''
            if sentiment == 'positive':
                sentiment_class = 'text-success'
            elif sentiment == 'negative':
                sentiment_class = 'text-danger'
            else:
                sentiment_class = 'text-warning'
                
            using_model = result.get('using_model', False)
            model_id = session.get('current_model_id')
            model_text = f"using custom model ({model_id[:8]}...)" if using_model and model_id else "using TextBlob"
            
            flash(f'Tweet: "{tweet_text[:50]}..." | Sentiment: <span class="{sentiment_class}">{sentiment.upper()}</span> | Polarity: {polarity:.2f} | Subjectivity: {subjectivity:.2f} | {model_text}', 'success')
        else:
            flash(f'Error analyzing tweet: {response.text}', 'error')
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
    
    return redirect(url_for('index', tab='single_tweet'))

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple tweets from a file."""
    
    if 'tweet_file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index', tab='batch_analysis'))
    
    tweet_file = request.files['tweet_file']
    
    if tweet_file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index', tab='batch_analysis'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(tweet_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        tweet_file.save(file_path)
        
        # Read the file (assuming CSV or TXT)
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            # Assuming the tweet text is in the first column
            tweets = df.iloc[:, 0].tolist()
        else:
            # Read as text file with one tweet per line
            with open(file_path, 'r', encoding='utf-8') as f:
                tweets = [line.strip() for line in f if line.strip()]
        
        # Limit to 100 tweets for demo purposes
        if len(tweets) > 100:
            flash(f'File contains {len(tweets)} tweets. Analyzing first 100 for performance reasons.', 'warning')
            tweets = tweets[:100]
        
        # Prepare data for API
        data = {
            'tweets': tweets
        }
        
        # Add model_id if available
        if session.get('current_model_id'):
            data['model_id'] = session.get('current_model_id')
        
        # Send to API
        response = requests.post(f"{API_BASE_URL}/analyze/batch", json=data)
        
        if response.status_code == 200:
            results = response.json()
            
            # Save results to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"sentiment_analysis_{timestamp}.csv"
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            
            # Create DataFrame from results
            result_data = []
            for item in results['results']:
                result_data.append({
                    'tweet': item['tweet'],
                    'sentiment': item['sentiment'],
                    'polarity': item.get('polarity', 'N/A'),
                    'subjectivity': item.get('subjectivity', 'N/A')
                })
            
            result_df = pd.DataFrame(result_data)
            result_df.to_csv(result_path, index=False)
            
            # Count sentiments
            sentiments = [item['sentiment'] for item in results['results']]
            positive_count = sentiments.count('positive')
            negative_count = sentiments.count('negative')
            neutral_count = sentiments.count('neutral')
            
            # Create summary
            summary = f"""
            Analysis complete for {len(tweets)} tweets:
            - Positive: {positive_count} ({positive_count/len(tweets)*100:.1f}%)
            - Negative: {negative_count} ({negative_count/len(tweets)*100:.1f}%)
            - Neutral: {neutral_count} ({neutral_count/len(tweets)*100:.1f}%)
            """
            
            flash(summary, 'success')
            
            # Return file for download
            return send_file(
                result_path,
                mimetype='text/csv',
                as_attachment=True,
                download_name=result_filename
            )
        else:
            flash(f'Error analyzing tweets: {response.text}', 'error')
            return redirect(url_for('index', tab='batch_analysis'))
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index', tab='batch_analysis'))

@app.route('/retrieve', methods=['POST'])
def retrieve_model():
    """Retrieve the current model."""
    
    if not session.get('current_model_id'):
        flash('No model uploaded yet', 'error')
        return redirect(url_for('index', tab='model_management'))
    
    try:
        response = requests.get(f"{API_BASE_URL}/model/{session.get('current_model_id')}")
        
        if response.status_code == 200:
            # Save the model
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"retrieved_model_{session.get('current_model_id')}.pkl")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Send the file to the user
            return send_file(output_path, as_attachment=True)
        else:
            flash(f'Error retrieving model: {response.text}', 'error')
            return redirect(url_for('index', tab='model_management'))
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index', tab='model_management'))

@app.route('/set_model_id', methods=['POST'])
def set_model_id():
    """Manually set a model ID."""
    
    model_id = request.form.get('model_id', '')
    
    if model_id:
        session['current_model_id'] = model_id
        flash(f'Model ID set to: {model_id}', 'success')
        # Redirect to the Single Tweet tab after setting model ID
        return redirect(url_for('index', tab='single_tweet'))
    else:
        flash('No model ID provided', 'error')
        return redirect(url_for('index', tab='model_management'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analysis (for AJAX calls)"""
    data = request.json
    
    if not data or 'tweet' not in data:
        return jsonify({"error": "No tweet provided"}), 400
    
    tweet_text = data['tweet']
    
    # Prepare data for API - use correct format
    api_data = {
        'tweet': tweet_text
    }
    
    # Add model_id if available
    if session.get('current_model_id'):
        api_data['model_id'] = session.get('current_model_id')
    
    # Send to API
    response = requests.post(f"{API_BASE_URL}/analyze", json=api_data)
    
    # Return the API response directly
    return response.json(), response.status_code

@app.route('/clear_model', methods=['POST'])
def clear_model():
    """Clear the current model ID."""
    if 'current_model_id' in session:
        session.pop('current_model_id')
        flash('Model ID cleared. Using default TextBlob analysis.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 80)))