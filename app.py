from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import base64
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Define the model class (same as training)
class GANDetectionModel(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(GANDetectionModel, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256 * 4 * 4, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load model
try:
    model = GANDetectionModel()
    model.load_state_dict(torch.load('models/trained_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("✅ Trained model loaded successfully!")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    model = None

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>GAN Face Detector</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white;
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            }
            h1 { 
                text-align: center; 
                color: #333;
                margin-bottom: 10px;
            }
            .upload-box { 
                border: 3px dashed #ddd; 
                padding: 40px; 
                text-align: center; 
                margin: 30px 0; 
                border-radius: 15px;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            .upload-box:hover {
                border-color: #667eea;
                background: #f8f9ff;
            }
            .result { 
                margin: 20px 0; 
                padding: 25px; 
                border-radius: 15px; 
                text-align: center;
                font-size: 1.2em;
            }
            .real { 
                background: #d4edda; 
                color: #155724; 
                border: 2px solid #c3e6cb;
            }
            .fake { 
                background: #f8d7da; 
                color: #721c24; 
                border: 2px solid #f5c6cb;
            }
            img { 
                max-width: 300px; 
                margin: 15px 0; 
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            input[type="file"] {
                margin: 15px 0;
            }
            button {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                font-size: 1.1em;
                cursor: pointer;
                transition: transform 0.2s ease;
            }
            button:hover {
                transform: scale(1.05);
            }
            .confidence {
                font-size: 1.1em;
                font-weight: bold;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 GAN Face Detector</h1>
            <p style="text-align: center; color: #666;">Upload a face image to detect if it's real or AI-generated</p>
            
            <div class="upload-box" onclick="document.getElementById('fileInput').click()">
                <form id="uploadForm">
                    <input type="file" id="fileInput" accept="image/*" hidden>
                    <div style="font-size: 4em;">📁</div>
                    <h3>Click to Upload Image</h3>
                    <p>PNG, JPG, JPEG files</p>
                    <button type="submit" style="margin-top: 20px;">Analyze Image</button>
                </form>
            </div>
            
            <div id="result"></div>
        </div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const file = document.getElementById('fileInput').files[0];
                if (!file) {
                    alert('Please select a file first!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<div style="text-align: center; padding: 20px;"><div style="font-size: 2em;">🔍</div><p>Analyzing image...<br><small>This may take a few seconds</small></p></div>';
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        const isReal = data.final_prediction === 'Real';
                        resultDiv.innerHTML = `
                            <div class="result ${isReal ? 'real' : 'fake'}">
                                <h2>${isReal ? '✅ REAL FACE' : '❌ AI-GENERATED'}</h2>
                                <div class="confidence">Confidence: ${(data.final_confidence * 100).toFixed(1)}%</div>
                                <img src="${data.image_preview}" alt="Uploaded image">
                                <p><small>Powered by Deep Learning Model</small></p>
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `<div style="background: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; text-align: center;">
                            <h3>❌ Error</h3>
                            <p>${data.error}</p>
                        </div>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div style="background: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; text-align: center;">
                        <h3>❌ Upload Failed</h3>
                        <p>${error.message}</p>
                    </div>`;
                }
            };
            
            // Drag and drop support
            const uploadBox = document.querySelector('.upload-box');
            uploadBox.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadBox.style.borderColor = '#667eea';
                uploadBox.style.background = '#f0f4ff';
            });
            
            uploadBox.addEventListener('dragleave', () => {
                uploadBox.style.borderColor = '#ddd';
                uploadBox.style.background = '';
            });
            
            uploadBox.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadBox.style.borderColor = '#ddd';
                uploadBox.style.background = '';
                
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    document.getElementById('fileInput').files = e.dataTransfer.files;
                    document.getElementById('uploadForm').dispatchEvent(new Event('submit'));
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded. Please train the model first.'})
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Save file temporarily
    import random
    filename = f"temp_{random.randint(1000, 9999)}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        print(f"🔍 Analyzing: {filename}")
        
        # Load and preprocess image
        image = Image.open(filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        is_fake = bool(prediction.item() == 1)
        confidence_score = float(confidence.item())
        
        result = {
            'final_prediction': 'GAN-generated' if is_fake else 'Real',
            'final_confidence': confidence_score,
            'success': True,
            'method': 'Trained CNN Model'
        }
        
        # Add image preview
        with open(filepath, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        result['image_preview'] = f"data:image/jpeg;base64,{img_data}"
        
        print(f"✅ Result: {result['final_prediction']} (confidence: {confidence_score:.3f})")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})
    
    finally:
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass

if __name__ == '__main__':
    print("🚀 Starting GAN Detection Server...")
    print("📱 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)