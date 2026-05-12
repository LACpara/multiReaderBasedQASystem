from functools import wraps
from time import sleep
from logging import getLogger

logger = getLogger(__file__)


def retry(retries: int = 3, delay: float = 1):
    """
    函数执行失败时，重试

    :param retries: 最大重试的次数
    :param delay: 每次重试的间隔时间，单位 秒
    :return:
    """

    # 校验重试的参数，参数值不正确时使用默认参数
    if retries < 1 or delay <= 0:
        retries = 3
        delay = 1

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 第一次正常执行不算重试次数，所以retries+1
            for i in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 检查重试次数
                    if i == retries:
                        logger.error(f"Error: {repr(e)}")
                        logger.error(f'"{func.__name__}()" 执行失败，已重试{retries}次')
                        break
                    else:
                        logger.warning(f"Error: {repr(e)}，{delay}秒后第[{i+1}/{retries}]次重试...")
                        sleep(delay)
        return wrapper
    return decorator