# spotify-analytics

# frontend
cd backend
python3 -m venv venv
source venv/bin/activate # or .\venv\Scripts\activate for Windows
pip install -r ../requirements.txt
flask run

# backend

npm create vite@latest frontend -- --template react
npm install

cd C:\Users\songs\Desktop\111\Misc code\spotify-analytics && python backend/jobs/daily_snapshot.py >> data/history/snapshot.log 2>&1