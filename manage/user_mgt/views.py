import json

from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from manage.user_mgt.models import *
from utils.logger_util import LoggerUtil

logger = LoggerUtil.setup_logger("user_mgt_logger")


@csrf_exempt
@require_http_methods(["POST"])
def login_api(request):
    try:
        data = json.loads(request.body)
        logger.info(data)
        username = data.get('username')
        password = data.get('password')
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'message': 'JSON解析错误'}, status=400)

    if not username or not password:
        return JsonResponse({
            'success': False,
            'message': '用户名和密码不能为空'
        }, status=400)
    try:
        user = User.objects.get(username=username)
        if not user.password == password:
            return JsonResponse({
                'success': False,
                'message': '密码错误'
            }, status=401)
    except User.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': '用户不存在'
        }, status=401)

    return JsonResponse({
        'success': True,
        'message': '登录成功',
        'id': user.id
    })

@csrf_exempt
@require_http_methods(["POST"])
def register_api(request):
    try:
        data = json.loads(request.body)
        logger.info(data)
        username = data.get('username')
        password = data.get('password')
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'message': 'JSON解析错误'}, status=400)

    if not username or not password:
        return JsonResponse({
            'success': False,
            'message': '用户名和密码不能为空'
        }, status=400)

    # User = get_user_model()
    if User.objects.filter(username=username).exists():
        return JsonResponse({
            'success': False,
            'message': '用户已存在'
        }, status=400)


    user = User.objects.create(
        username=username,
        password=password,
        role_id=6,
        status=1
    )

    return JsonResponse({
        'success': True,
        'message': '注册成功'
    })

@csrf_exempt
@require_http_methods(["POST"])
def upload_app_key(request):
    try:
        data = json.loads(request.body)
        logger.info(data)
        username = data.get('username')
        llm = data.get('llm')
        app_key = data.get('app_key')
        user = User.objects.get(username=username)
        llmmodel = LLMModel.objects.get(username=username, llm=llm)
        if LLMModel.objects.filter(username=username).exists():
            llmmodel.delete(username=username, llm=llm)
        LLMModel.objects.create(
            user_id=user.id,
            llm=llm,
            app_key=app_key,
            status=1
        )
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'message': 'JSON解析错误'}, status=400)
    return JsonResponse({
        'success': True,
        'message': '注册成功'
    })


@csrf_exempt
@require_http_methods(["POST"])
def get_app_key(request):
    try:
        data = json.loads(request.body)
        logger.info(data)
        username = data.get('username')
        user = User.objects.filter(username=username).exists()
        # 通过user.id 查询LLMModel表里的app_key列表返回给前端
        app_key_list = list(LLMModel.objects.filter(user_id=user.id).values('llm', 'app_key'))
        return JsonResponse({
            'success': True,
            'message': '查询成功',
            'app_key_list': app_key_list
        })
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'message': 'JSON解析错误'}, status=400)
    return JsonResponse({
        'success': True,
        'message': '注册成功'
    })
