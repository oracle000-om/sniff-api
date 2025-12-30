# 🐾 Sniff - Reuniting Pets with Their Families

Sniff uses AI-powered facial recognition to help lost pets find their way home. Animal shelters and good samaritans can register found pets, and owners can search to find potential matches.

**Sniff will now and forever remain free to use for all.**

🌐 **Live at:** [sniffhome.org](https://sniffhome.org)

---

## ✨ Features

### For Pet Owners

- 🔍 **AI-Powered Search** - Upload a photo of your pet and instantly search the database
- 📊 **Match Confidence** - See similarity scores for potential matches
- 🏷️ **Claim System** - Claim your pet when you find a match
- 📱 **Mobile Responsive** - Optimized for all devices

### For Shelters & Finders

- 📸 **Dual Registration Paths** - Shelter intake or Good Samaritan reports
- 🏥 **Shelter Hero Mode** - Track microchips, intake names, and organization info
- 🦸 **Good Samaritan Mode** - Report found pets with location and holding status
- 📍 **Location Tracking** - GPS auto-fill for where pets were found
- 🔒 **Privacy First** - Contact info protected until claims are made

### Community & Support

- 💬 **Say Hi Page** - User feedback, developer contributions, and partnership inquiries
- 🤝 **Ways to Help** - Spread awareness, contribute code, or support the mission
- 📥 **Resource Downloads** - Flyers and media kits for community outreach

### Technical Features

- 🧠 **Deep Learning** - 2048-dimension facial embeddings for high accuracy
- ⚡ **Fast Search** - Vector similarity search with Milvus
- 🔐 **Privacy Compliant** - Hashed IPs, localStorage tracking
- 📊 **Real-time Stats** - Live pet registration counter
- 🎨 **Modern UI** - Clean, accessible interface with mobile-first design
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
- Responsive CSS with clamp() scaling
- Mobile-first design
- localStorage for claim tracking

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
git clone https://github.com/oracle000-om/sniff-api.git
cd sniff-api
```

2. **Create virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt --break-system-packages
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

**Shelter Path:**

1. Select "🏥 Shelter Hero"
2. Upload clear photo of the pet's face
3. Enter intake name, species, microchip (if available)
4. Select or add your shelter/organization
5. Add any behavioral notes
6. Submit registration

**Good Samaritan Path:**

1. Select "🦸 Good Samaritan"
2. Upload clear photo of the pet's face
3. Enter name from tag (if visible), species, microchip (if checked)
4. Select if you're holding the pet or just spotted them
5. Enter location found (or use GPS auto-fill)
6. Add contact info (required if holding the pet)
7. Submit registration

### Search for Your Lost Pet

1. Navigate to "Find Your Lost Pet" card
2. Upload a clear photo of your pet
3. View potential matches with confidence scores
4. Review match details (species, location, finder info)
5. Claim your pet if you find a match
6. Contact info revealed after claiming

### Claim System

- Claims tracked via localStorage (privacy-first approach)
- Each browser can claim each pet once
- Warning message reminds users to only claim genuine matches
- Color-coded badges show claim activity:
  - No badge: Unclaimed
  - 🟡 Yellow: 1 claimer
  - 🟠 Orange: 2-3 claimers
  - 🔴 Red: 4-5 claimers
  - 🚨 Dark Red: 6-9 claimers
  - 🚫 Alert: 10+ claimers (contact directly)

**Claim Validation:**

- Confirmation dialog warns against false claims
- Rate limiting: 5 claims per IP per hour
- Tooltips explain high claim counts

---

## 📄 Pages

### Home (`/`)

- Dual registration form (shelter/finder)
- Pet search and matching
- Live stats counter

### Ways to Help (`/ways-to-help`)

- Download flyers and media kits
- Find local shelters
- Support via Ko-fi
- GitHub repository and contributions

### Say Hi (`/say-hi`)

- **I'm a user** - General feedback form
- **I'm a developer** - GitHub issues link
- **I want to partner** - Partnership inquiry form

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

Response includes:

- Database connection status
- Total pets registered
- Images stored
- Disk space available

---

## 🚀 Deployment

### Prerequisites

- Milvus running (Docker or cloud)
- Python 3.11+ environment
- Static file serving configured

### Production Checklist

- [ ] Change claim salt in `app.py`
- [ ] Set up environment variables
- [ ] Configure HTTPS/SSL
- [ ] Enable rate limiting
- [ ] Set up automated backups
- [ ] Add monitoring/analytics
- [ ] Update contact email in say-hi page

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Open Pull Request

### Ways to Contribute

- **Code:** Bug fixes, features, optimizations
- **Design:** UI/UX improvements, mobile optimization
- **Content:** Flyers, social media graphics, translations
- **Outreach:** Share with shelters, spread awareness
- **Support:** Ko-fi donations for server costs

### Development Guidelines

- Follow PEP 8 style guide
- Add comments for complex logic
- Test on mobile devices
- Update documentation
- Keep commits atomic and descriptive

---

## 📁 Project Structure

```
sniff-api/
├── app.py                    # Main FastAPI application
├── database.py               # Milvus connection handler
├── models/
│   ├── matching.py           # Pet matching logic
│   └── quality_check.py      # Image quality checker
├── templates/
│   ├── index.html            # Main page
│   ├── ways-to-help.html     # Community support page
│   └── say-hi.html           # Feedback & partnerships
├── static/
│   └── sniff_flyer.pdf       # Downloadable flyer
├── data/
│   ├── images/               # Uploaded pet photos
│   └── claims.json           # Claim tracking
├── backups/                  # Database backups
├── docker-compose.yml        # Docker services
├── backup_database.py        # Backup script
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🔒 Privacy & Security

- ✅ **IP Hashing** - User IPs hashed with salt
- ✅ **No Personal Data** - Only hashed identifiers stored
- ✅ **Protected Contacts** - Finder info only shown after claim
- ✅ **Rate Limiting** - 5 claims per IP per hour
- ✅ **localStorage Tracking** - Privacy-first claim tracking
- ✅ **Secure Storage** - Docker volumes for persistence
- ✅ **Input Validation** - File type and size checks

**Note:** For production, implement additional security:

- HTTPS/SSL certificates
- API rate limiting
- Input sanitization
- Regular security audits
- GDPR compliance measures

---

## 📝 License

This project is open source and available under the MIT License.

**Free Forever:** Sniff will always remain free to use for all shelters, rescues, and pet owners.

---

## 🙏 Acknowledgments

- **DeepFace** - Face recognition library
- **Milvus** - Vector database
- **FastAPI** - Web framework
- **All contributors** - Thank you for helping reunite pets with their families

---

## 📧 Contact

- **Website:** [sniffhome.org](https://sniffhome.org)
- **Feedback:** [Say Hi Page](https://sniffhome.org/say-hi)
- **Issues:** [GitHub Issues](https://github.com/oracle000-om/sniff-api/issues)
- **Email:** enter@daye.town

---

## 🗺️ Roadmap

**v2.0 (Current):**

- ✅ Dual registration paths (shelter/finder)
- ✅ Mobile-responsive design
- ✅ Community pages (ways-to-help, say-hi)
- ✅ Enhanced claim system with validation
- ✅ GPS location auto-fill
- ✅ Partnership inquiry system

**Future:**

- 🔄 Email notifications for matches
- 🔄 Multi-language support
- 🔄 Advanced filtering (by location, species, date)
- 🔄 Success stories showcase
- 🔄 Shelter dashboard analytics

---

**Built with ❤️ for the pets who love us, in honor of Henry**

🐾 Together, we bring our buddies home.
