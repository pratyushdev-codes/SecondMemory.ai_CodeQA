from apscheduler.schedulers.background import BackgroundScheduler
import datetime
from .tasks import   trackVectorDBList, flushVectorDB

scheduler = BackgroundScheduler()




# Schedule the job to run once at startup
scheduler.add_job(flushVectorDB, trigger="date", run_date=datetime.datetime.now())

# Schedule the job to run every 30 secs
scheduler.add_job(trackVectorDBList, 'interval', seconds=30)