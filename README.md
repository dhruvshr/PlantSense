# PlantSense 🌿

PlantSense is an AI-powered plant disease detection and diagnosis system that helps users identify and treat plant diseases through image analysis and interactive chat-based insights.

## Features

- **Real-time Plant Disease Detection**: Upload images of your plants for instant disease detection
- **Interactive Chat Interface**: Get detailed insights and treatment recommendations through a conversational AI
- **Image History**: Access your previously uploaded images and their diagnoses
- **High Accuracy**: Powered by a fine-tuned ResNet model achieving 92%+ accuracy
- **Responsive Design**: Clean, modern interface that works across devices

## Technology Stack

### Backend
- Flask (Web Framework)
- SQLAlchemy (ORM)
- PyTorch (Deep Learning)
- OpenAI API (Chat Insights)
- SQLite (Database)

### Frontend
- HTML5/CSS3
- JavaScript
- Jinja2 Templates

### ML/AI
- ResNet18 (Pre-trained Model)
- Custom Training Pipeline
- Plant Village Dataset

## Project Structure

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/plantsense.git
cd plantsense
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

5. Initialize the database:
```bash
flask db upgrade
```

## Usage

1. Start the Flask development server:
```bash
python run.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload a plant image through the interface

4. View the diagnosis and chat with PlantSense for detailed insights

## Model Training

The project uses a custom-trained ResNet18 model for plant disease detection. To train the model:

1. Download the Plant Village dataset
2. Run the training script:
```bash
python scripts/train.py
```

Training configurations can be modified in `src/training/trainer.py`.

## Development

### Database Migrations

```bash
flask db migrate -m "Migration message"
flask db upgrade
```

### Adding New Features

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes
3. Run tests
4. Submit a pull request

## API Reference

### Image Upload Endpoint
- POST `/`: Upload plant images for analysis
- Response: Redirects to chat interface with diagnosis

### Chat Endpoint
- POST `/chat`: Submit user queries
- Response: AI-generated insights and recommendations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- Plant Village Dataset for training data
- OpenAI for chat capabilities
- PyTorch team for the deep learning framework
- Flask team for the web framework

## Contact

For questions and support, please open an issue on the GitHub repository.