import os
from django.db import models
from ai.settings.settings import BASE_DIR


def upload_to(instance, filename):
    print(BASE_DIR)
    return os.path.join(BASE_DIR, 'static', 'images', filename).format(filename=filename)


class Requested(models.Model):
    user = models.CharField(null=False, max_length=50)
    image = models.ImageField(verbose_name='image',
                              null=True, blank=True, upload_to=upload_to)
    # display_name = models.CharField('상품명(노출용)', max_length=100)
    # price = models.PositiveIntegerField('권장판매가')
    # sale_price = models.PositiveIntegerField('실제판매가')
    # code = models.CharField(max_length=8, default=generate_unique_code, unique=True)
    # writer = models.CharField(User, max_length=50, null=True)
    # listing_or_not = models.BooleanField(null=False, default=False) # 장바구니 표현하려 했으나, boolean이면 안될 거 같아서 판매가능여부로 치자
    # like_count = models.IntegerField(null=True, default=0)
    # created_at = models.DateTimeField(auto_now_add=True)
    # update_date = models.DateTimeField(auto_now=True)
    # is_deleted = models.BooleanField('삭제여부', default=False)
    # delete_date = models.DateTimeField('삭제날짜', null=True, blank=True)

    # current_song = models.CharField(max_length=50, null=True)
    # category = models.ForeignKey(ProductCategory, on_delete=models.DO_NOTHING)
    # market = models.ForeignKey(Market, on_delete=models.DO_NOTHING)

    # is_hidden = models.BooleanField('노출여부', default=False)
    # is_sold_out = models.BooleanField('품절여부', default=False)

    # hit_count = models.PositiveIntegerField('조회수', default=0)
    # review_count = models.PositiveIntegerField('리뷰수', default=0)
    # review_point = models.PositiveIntegerField('리뷰평점', default=0)
    # questions = GenericRelation(Question, related_query_name="question")


class Responsed(models.Model):
    user = models.CharField(null=False, max_length=50)
    result = models.CharField(null=False, max_length=500)
