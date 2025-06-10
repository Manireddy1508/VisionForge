# AI Image Generation Playground

A powerful AI-powered image generation system that combines advanced prompt engineering, image analysis, and vector search capabilities to create high-quality images. The system uses a combination of GPT-4 Vision for image analysis, OpenAI's DALL-E for image generation, and Milvus for vector similarity search.

## 🏗️ System Architecture

The system consists of several key components:

### Core Components

1. **Prompt Enhancement System**
   - `PromptEnhancer`: Enhances user prompts using GPT-4
   - `PromptRouter`: Routes prompts to appropriate models
   - `PromptEditor`: Allows manual prompt editing
   - `ImageDescriber`: Analyzes reference images using BLIP and GPT-4 Vision

2. **Vector Database (Milvus)**
   - Stores and searches similar prompts and images
   - Enables prompt enhancement using historical data
   - Maintains collections for prompts and images

3. **Image Generation**
   - Uses OpenAI's DALL-E for high-quality image generation
   - Supports multiple output variations
   - Configurable parameters (steps, guidance, seed)

4. **Web Interface**
   - Gradio-based main application (port 7860)
   - Streamlit-based dashboard (port 7861)
   - Interactive prompt editing and image generation

### Infrastructure

- **Docker-based Deployment**
  - Multi-container setup with Docker Compose
  - Services: App, Dashboard, Milvus, MinIO, etcd
  - Volume management for persistent storage

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.12+
- OpenAI API key
- Google Cloud credentials (for cloud deployment)

### Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_IMAGE_MODEL=dall-e-3
GOOGLE_CLOUD_PROJECT=your_gcp_project
GCS_BUCKET_NAME=your_bucket_name
```

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Manireddy1508/newcrux.git
   cd newcrux
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

3. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

4. **Access the applications:**
   - Main app: http://localhost:7860
   - Dashboard: http://localhost:7861

### Docker Services

The system runs several Docker services:

- **app**: Main Gradio application (port 7860)
- **dashboard**: Streamlit dashboard (port 7861)
- **standalone**: Milvus vector database
- **minio**: Object storage for Milvus
- **etcd**: Distributed key-value store for Milvus

## 📁 Project Structure

```
.
├── src/
│   ├── app.py                 # Main Gradio application
│   ├── image_generator.py     # Image generation logic
│   ├── milvus_utils.py       # Vector database operations
│   ├── dashboard/            # Streamlit dashboard
│   └── prompting/            # Prompt engineering modules
│       ├── prompt_enhancer.py
│       ├── prompt_router.py
│       ├── prompt_editor.py
│       ├── image_describer.py
│       └── model_manager.py
├── tests/                    # Test suite
├── docs/                     # Documentation
├── volumes/                  # Persistent storage
├── Dockerfile               # Main application container
└── docker-compose.yml       # Multi-container setup
```

## 🔧 Key Features

1. **Advanced Prompt Engineering**
   - GPT-4 powered prompt enhancement
   - Reference image analysis
   - Similar prompt search and reuse

2. **Image Generation**
   - Multiple output variations
   - Configurable parameters
   - Reference image support

3. **Vector Search**
   - Similar prompt retrieval
   - Image similarity search
   - Historical data reuse

4. **Monitoring and Analytics**
   - Streamlit dashboard
   - Operation logging
   - Performance metrics

## 🛠️ Development

### Adding New Features

1. **New Models**
   - Add model configuration in `src/prompting/model_manager.py`
   - Update `PromptRouter` to handle new model types

2. **New Prompt Templates**
   - Add templates in `src/prompting/prompt_templates.py`
   - Update `PromptEnhancer` to use new templates

3. **New Vector Search Features**
   - Extend `milvus_utils.py` with new search functions
   - Update collections schema if needed

### Testing

Run tests with:
```bash
pytest tests/
```

## 📊 Monitoring

The Streamlit dashboard (port 7861) provides:
- System health metrics
- Operation logs
- Performance statistics
- Resource usage

## 🔐 Security

- API keys stored in environment variables
- Docker container isolation
- Secure volume mounting
- No sensitive data in version control

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Manireddy1508 - Initial work

## 🙏 Acknowledgments

- OpenAI for GPT-4 and DALL-E
- Milvus for vector database
- Gradio and Streamlit for UI
- All contributors and users 