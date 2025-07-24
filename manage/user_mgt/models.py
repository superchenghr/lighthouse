from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import AbstractUser
from django.db import models


class Role(models.Model):
    STATUS_CHOICES = ((1, '启用'), (0, '禁用'))

    name = models.CharField(max_length=100, unique=True, verbose_name='角色名称')
    description = models.TextField(null=True, blank=True, verbose_name='角色描述')
    permissions = models.JSONField(null=True, blank=True, verbose_name='权限配置JSON')
    status = models.SmallIntegerField(choices=STATUS_CHOICES, default=1, verbose_name='状态')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lh_role'  # 显式指定数据库表名
        indexes = [models.Index(fields=['status'], name='idx_role_status')]
        verbose_name = '角色'

    def __str__(self):
        return self.name


class User(models.Model):
    STATUS_CHOICES = ((1, '启用'), (0, '禁用'))
    username = models.CharField(max_length=255, unique=True, verbose_name='用户名')
    password = models.CharField(max_length=255, verbose_name='密码')  # 实际项目应使用AbstractBaseUser密码加密
    phone = models.CharField(max_length=20, null=True, blank=True, verbose_name='手机号')
    role = models.ForeignKey(Role, on_delete=models.PROTECT, db_index=True, verbose_name='角色', default=0)  # 外键+索引[6,8](@ref)
    status = models.SmallIntegerField(choices=STATUS_CHOICES, default=1, verbose_name='状态')
    last_login_at = models.DateTimeField(blank=True, null=True)  # 显式映射到数据库字段    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lh_user'
        indexes = [
            models.Index(fields=['status'], name='idx_user_status'),
            models.Index(fields=['role'], name='idx_role_id')
        ]
        verbose_name = '用户'

    def __str__(self):
        return self.username


class LLMModel(models.Model):
    STATUS_CHOICES = ((1, '启用'), (0, '禁用'))

    llm = models.CharField(max_length=255, db_index=True, verbose_name='模型标识')  # 带索引的字段[10](@ref)
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_index=True, verbose_name='所属用户', default=0)
    app_key = models.CharField(max_length=255, verbose_name='应用密钥')
    status = models.SmallIntegerField(choices=STATUS_CHOICES, default=1, verbose_name='状态')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'lh_llm_model'
        indexes = [
            models.Index(fields=['llm'], name='idx_llm'),
            models.Index(fields=['user'], name='idx_user_id')
        ]
        verbose_name = '大语言模型配置'

    def __str__(self):
        return f"{self.llm} (用户:{self.user_id})"