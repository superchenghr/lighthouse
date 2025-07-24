import pymysql
pymysql.install_as_MySQLdb()

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-d@o%v)9%#12_8or1o#9h$d79i+dp@!+k5t6dg_r9b_rvgwnpiy'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['*']

ROOT_URLCONF = 'manage.urls'


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'lighthouse',
        'USER': 'root',
        'PASSWORD': '',
        'HOST': 'localhost',
        'PORT': '3306',
        'OPTIONS': {'charset': 'utf8mb4'},
    }
}

INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'corsheaders',
    'manage.user_mgt',
    'django_extensions',
]

# AUTH_USER_MODEL = 'user_mgt.User'

APPEND_SLASH = False

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

# 允许所有域（仅开发阶段这样设置）
CORS_ALLOW_ALL_ORIGINS = True

LANGUAGE_CODE = 'zh-hans'  # 中文
TIME_ZONE = 'Asia/Shanghai'  # 东八区
USE_I18N = True
USE_TZ = False