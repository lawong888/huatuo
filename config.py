import os

# Local-testing defaults only: setdefault() lets real env vars (e.g. on Render) win.

os.environ.setdefault('REDIS_HOST', 'redis-14594.c295.ap-southeast-1-1.ec2.redns.redis-cloud.com')
os.environ.setdefault('REDIS_PORT', '14594')
os.environ.setdefault('REDIS_USERNAME', 'default')
os.environ.setdefault('REDIS_PASSWORD', '1vQuf5Jn5OtUmK5fISf1BYaxynFRsesf')
os.environ.setdefault('SPEECH_KEY', 'd59e530807384648a02e89407fd11bb1')
os.environ.setdefault('SPEECH_REGION', 'southeastasia')

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
