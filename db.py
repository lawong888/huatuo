import os
import redis
import tiktoken
import datetime

db = redis.Redis(
          host='redis-11731.c1.ap-southeast-1-1.ec2.cloud.redislabs.com',
          port=11731,
          username='default',
          password='2bDDtX5SOWQ3W2lZVhI6IQKsAWtacETL',
          decode_responses=True
          )

# length of user list in db
user_db = db.llen('user_message')

    
# push data to redis
def db_push(transcript_text, chat_transcript, entry_datetime, token):

    db.rpush('user_message', transcript_text)
    db.rpush('matthew_response', chat_transcript)
    db.rpush('Datetime_HK', entry_datetime)
    db.rpush('Token_HK', token)
