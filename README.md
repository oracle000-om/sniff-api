# 🐾 Sniff - Reuniting Pets with Their Families

Sniff uses AI-powered facial recognition to help lost pets find their way home. Animal shelters and good samaritans can register found pets, and owners can search to find potential matches.

Sniff will now and forever remain free to use for all.

🌐 **Live at:** [sniffhome.org](https://sniffhome.org)

---

## ✨ Features

### For Pet Owners
- 🔍 **AI-Powered Search** - Upload a photo of your pet and instantly search the database
- 📊 **Match Confidence** - See similarity scores for potential matches
- 🏷️ **Claim System** - Claim your pet when you find a match
- 📱 **Mobile Friendly** - Works on all devices

### For Shelters & Finders
- 📸 **Easy Registration** - Upload photos of found pets with details
- 🔒 **Privacy First** - Contact info protected until claims are made
- 📈 **Real-Time Stats** - Track total pets and successful reunions
- 🤝 **Community Driven** - Good Samaritans can also register found pets

### Technical Features
- 🧠 **Deep Learning** - 2048-dimension facial embeddings for high accuracy
- ⚡ **Fast Search** - Vector similarity search with Milvus
- 🔐 **Privacy Compliant** - Hashed IPs, GDPR-ready
- 📊 **Analytics** - Track searches, registrations, and claims
- 💾 **Auto Backups** - Daily database backups with 7-day retention
- 🐳 **Production Ready** - Dockerized Milvus for scalability

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI (Python web framework)
- Milvus (Vector database)
- DeepFace (Facial recognition)
- OpenCV (Image processing)

**Frontend:**
- Vanilla JavaScript
- Responsive CSS
- Mobile-first design

**Infrastructure:**
- Docker & Docker Compose
- Railway (Deployment)
- GitHub (Version control)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- Git

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/sniff-api.git
cd sniff-api
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Start Docker Milvus:**
```bash
docker-compose up -d
```

Wait 60 seconds for services to start.

5. **Run the application:**
```bash
uvicorn app:app --reload
```

6. **Open browser:**
```
http://localhost:8000
```

---

## 🐳 Docker Services

The application uses three Docker containers:

- **milvus-standalone** - Vector database for pet embeddings
- **milvus-etcd** - Metadata storage
- **milvus-minio** - Object storage for Milvus

**Check status:**
```bash
docker ps
```

**Stop services:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f
```

---

## 📖 Usage

### Register a Found Pet

1. Select role: Shelter or Good Samaritan
2. Upload clear photo of the pet's face
3. Fill in details (name, species, location, contact)
4. Submit registration

### Search for Your Lost Pet

1. Upload a clear photo of your pet
2. View potential matches with confidence scores
3. Claim your pet if you find a match
4. Contact info revealed after claiming

### Claim Rules

- Each person can claim each pet once
- Maximum 10 different pets per user
- Claims tracked by IP (privacy-protected with hashing)
- Color-coded badges show claim activity:
  - 🟡 Yellow: 1 claimer
  - 🟠 Orange: 2-3 claimers
  - 🔴 Red: 4-5 claimers
  - 🚨 Dark Red: 6-9 claimers
  - 🚫 Max: 10+ claimers (contact directly)

---

## 💾 Backup & Restore

### Automatic Backups

Daily backups run at 2 AM (if cron configured):
```bash
# Set up cron job
crontab -e

# Add this line:
0 2 * * * cd /path/to/sniff-api && /usr/bin/python3 backup_database.py >> backup.log 2>&1
```

### Manual Backup
```bash
python3 backup_database.py
```

Backups stored in `backups/` directory (last 7 kept).

### Restore from Backup

See `RESTORE.md` for detailed instructions.

Quick restore:
```bash
# Stop app first
docker-compose down

# Restore from backup
cp backups/backup_YYYYMMDD_HHMMSS/milvus_demo.db ./
cp -r backups/backup_YYYYMMDD_HHMMSS/images/* data/images/
cp backups/backup_YYYYMMDD_HHMMSS/claims.json data/

# Restart
docker-compose up -d
uvicorn app:app --reload
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file for production:
```bash
PORT=8000
MILVUS_HOST=localhost
MILVUS_PORT=19530
SALT_SECRET=your-random-salt-here
```

### Change Claim Salt (Production)

⚠️ **Important:** Change the salt in `app.py` before production:
```python
# In /api/v1/claim endpoint
salt = "your-unique-random-salt-here"  # Change this!
```

Generate random salt:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 📊 Health Check

Monitor application health:
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "database": "connected",
  "pets_registered": 42,
  "images_stored": 42,
  "data_synced": true,
  "disk_free_gb": 120.5,
  "timestamp": "2025-10-24T12:00:00"
}
```

---

## 🚀 Deployment (Railway)

1. **Create Railway account:** https://railway.app
2. **Connect GitHub repo**
3. **Add environment variables**
4. **Deploy!**

Detailed deployment guide: See `DEPLOY.md` (if created)

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Open Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

## 📁 Project Structure
```
sniff-api/
├── app.py                 # Main FastAPI application
├── models/
│   ├── matching.py        # Pet matching logic
│   └── quality_check.py   # Image quality checker
├── templates/
│   └── index.html         # Frontend interface
├── data/
│   ├── images/            # Uploaded pet photos
│   └── claims.json        # Claim tracking
├── backups/               # Database backups
├── docker-compose.yml     # Docker services
├── backup_database.py     # Backup script
├── migrate_to_docker.py   # Migration script
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🔒 Privacy & Security

- ✅ **IP Hashing** - User IPs hashed with salt (GDPR compliant)
- ✅ **No Personal Data** - Only hashed identifiers stored
- ✅ **Protected Contacts** - Finder info only shown after claim
- ✅ **Rate Limiting** - Claim limits prevent spam
- ✅ **Secure Storage** - Docker volumes for persistence

**Note:** For production, implement additional security:
- HTTPS/SSL certificates
- Rate limiting on API endpoints
- Input validation and sanitization
- Regular security audits

---

## 📝 License

This project is open source and available under the MIT License.

**Free Forever:** Sniff will always remain free to use for all shelters, rescues, and pet owners.

---

## 🙏 Acknowledgments

- **DeepFace** - Face recognition library
- **Milvus** - Vector database
- **FastAPI** - Web framework

---

## 📧 Contact

- **Website:** [sniffhome.org](https://sniffhome.org)
- **Issues:** [GitHub Issues](https://github.com/oracle000-om/sniff-api/issues)
- **Email:** support@sniffhome.org (coming soon)

---

## 🗺️ Roadmap

**Phase 1 (Launched):**
- ✅ Basic facial recognition
- ✅ Shelter registration
- ✅ Search and claim system
- ✅ Docker deployment

---

**Built with ❤️ for the pets who love us, in honor of Henry**

🐾 Together, we bring them home.