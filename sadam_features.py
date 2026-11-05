"""Online weather, live web lookup, and voice-turn routing for Sadam.

Python 3.6+ compatible; only dependency is the project's existing requests.
No Jetson-specific imports so these functions can be unit-tested offline.
"""
import os
import re
import time
import requests

DEFAULT_CITY = os.environ.get('SADAM_DEFAULT_CITY', 'Astana')
WEB_SEARCH_MODEL = os.environ.get('SADAM_WEB_MODEL', 'gpt-5-search-api')
WEB_ENABLED = os.environ.get('SADAM_WEB_ENABLED', '1') == '1'
WEATHER_ENABLED = os.environ.get('SADAM_WEATHER_ENABLED', '1') == '1'

WEATHER_WORDS = re.compile(
    r'(?i)(?:\bweather\b|\bforecast\b|\btemperature\b|'
    r'погод\w*|температур\w*|ауа рай\w*|күн рай\w*|'
    r'الطقس|الجو|حرارة|درجة الحرارة)'
)
RECENT_WORDS = re.compile(
    r'(?i)(?:\b(?:today|yesterday|tomorrow|latest|current|recent|'
    r'now|breaking|news|election|elections|results|score|winner|'
    r'president|prime minister|stock price|exchange rate|live|'
    r'who won|who is the)\b|'
    r'сегодня|вчера|завтра|сейчас|последн\w*|свеж\w*|новост\w*|'
    r'выбор\w*|результат\w*|побед\w*|президент\w*|'
    r'бүгін|кеше|ертең|жаңалық\w*|сайлау\w*|'
    r'النهارده|امبارح|أمس|اليوم|بكره|الأخبار|انتخابات|فاز|النتيجة|الرئيس)'
)

CONDITIONS = {
    0: ('ясно', 'clear', 'ашық', 'صافي'),
    1: ('преимущественно ясно', 'mostly clear', 'көбіне ашық', 'صافي في الغالب'),
    2: ('переменная облачность', 'partly cloudy', 'ала бұлтты', 'غائم جزئياً'),
    3: ('пасмурно', 'overcast', 'бұлтты', 'غائم'),
    45: ('туман', 'fog', 'тұман', 'شبورة'),
    48: ('изморозь и туман', 'freezing fog', 'аязды тұман', 'شبورة متجمدة'),
    51: ('небольшая морось', 'light drizzle', 'сіркіреме жаңбыр', 'رذاذ خفيف'),
    53: ('морось', 'drizzle', 'сіркіреме жаңбыр', 'رذاذ'),
    55: ('сильная морось', 'dense drizzle', 'қатты сіркіреме', 'رذاذ كثيف'),
    61: ('небольшой дождь', 'light rain', 'аздаған жаңбыр', 'مطر خفيف'),
    63: ('дождь', 'rain', 'жаңбыр', 'مطر'),
    65: ('сильный дождь', 'heavy rain', 'қатты жаңбыр', 'مطر شديد'),
    71: ('небольшой снег', 'light snow', 'аздаған қар', 'تلج خفيف'),
    73: ('снег', 'snow', 'қар', 'تلج'),
    75: ('сильный снег', 'heavy snow', 'қалың қар', 'تلج شديد'),
    80: ('ливень', 'rain showers', 'нөсер жаңбыр', 'زخات مطر'),
    81: ('ливень', 'rain showers', 'нөсер жаңбыр', 'زخات مطر'),
    82: ('сильный ливень', 'heavy rain showers', 'қатты нөсер', 'زخات مطر شديدة'),
    95: ('гроза', 'thunderstorm', 'найзағай', 'عاصفة رعدية'),
    96: ('гроза с градом', 'thunderstorm with hail', 'бұршақты найзағай', 'رعد وبرد'),
    99: ('сильная гроза с градом', 'severe thunderstorm with hail', 'қатты найзағай', 'رعد وبرد شديد'),
}


def weather_intent(question):
    return bool(WEATHER_WORDS.search(question or ''))


def weather_city(question, default_city=None):
    """Extract a common spoken location; use explicit default otherwise."""
    default = default_city or DEFAULT_CITY
    # Allow short city names in Latin/Cyrillic and Arabic, but never URLs.
    match = re.search(
        r'(?i)(?:\b(?:in|for|at)\s+|\b(?:в|во|для)\s+|'
        r'\b(?:في|بمدينة)\s+|\b(?:қаласында|қаласындағы)\s+)'
        r'([A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі\u0600-\u06FF\- ]{2,45})',
        question or ''
    )
    if not match:
        return default
    value = match.group(1).strip(' ,.!?،؟')
    # The speaker may append "كام" ("how much") after the city.
    value = re.split(
        r'(?i)\b(?:today|tomorrow|now|right now|please|currently|'
        r'сегодня|завтра|сейчас|пожалуйста|бүгін|ертең|'
        r'اليوم|بكره|دلوقتي|حاليا|كام|قد|كم|إيه|ايه)\b',
        value, maxsplit=1
    )[0].strip(' ,.!?،؟')
    aliases = {
        'астане': 'Astana', 'астана': 'Astana', 'астану': 'Astana',
        'астананың': 'Astana', 'астанада': 'Astana',
        'أستانا': 'Astana', 'استانا': 'Astana',
        'алматы': 'Almaty', 'алмате': 'Almaty', 'алматыда': 'Almaty'
    }
    return aliases.get(value.casefold(), value or default)

def wants_tomorrow(question):
    return bool(re.search(r'(?i)\b(?:tomorrow|завтра|ертең|بكره|غدا)\b', question or ''))


def online_search_intent(question, recent_context=False):
    if weather_intent(question):
        return False
    if RECENT_WORDS.search(question or ''):
        return True
    # Follow-up on fresh news, e.g. "what about the Senate?"
    return recent_context and bool((question or '').strip())


def _weather_description(code, language):
    index = {'Russian': 0, 'English': 1, 'Kazakh': 2, 'Egyptian Arabic': 3}.get(language, 1)
    return CONDITIONS.get(code, ('необычная погода', 'unusual conditions', 'ерекше ауа райы', 'جو غير معتاد'))[index]


def get_weather(question, language='Russian', session=requests):
    """Return a source-grounded spoken answer without another LLM request."""
    city = weather_city(question)
    geo = session.get(
        'https://geocoding-api.open-meteo.com/v1/search',
        params={'name': city, 'count': 1, 'language': 'en', 'format': 'json'},
        timeout=(5, 12)
    )
    geo.raise_for_status()
    candidates = geo.json().get('results') or []
    if not candidates:
        raise ValueError('Weather city not found: ' + city)
    place = candidates[0]
    city_name = place.get('name', city)
    forecast = session.get(
        'https://api.open-meteo.com/v1/forecast',
        params={
            'latitude': place['latitude'], 'longitude': place['longitude'],
            'current': 'temperature_2m,apparent_temperature,weather_code,wind_speed_10m',
            'daily': 'temperature_2m_max,temperature_2m_min,weather_code',
            'forecast_days': 2, 'timezone': 'auto'
        }, timeout=(5, 12)
    )
    forecast.raise_for_status()
    data = forecast.json()
    tomorrow = wants_tomorrow(question)
    if tomorrow:
        daily = data['daily']
        temp_min = round(daily['temperature_2m_min'][1])
        temp_max = round(daily['temperature_2m_max'][1])
        code = daily['weather_code'][1]
        condition = _weather_description(code, language)
        if language == 'Russian':
            answer = 'Завтра в {}: {}, от {} до {} °C.'.format(city_name, condition, temp_min, temp_max)
        elif language == 'Kazakh':
            answer = 'Ертең {} қаласында {}, {} пен {} °C аралығында.'.format(city_name, condition, temp_min, temp_max)
        elif language == 'Egyptian Arabic':
            answer = 'بكرة في {} الجو {}، من {} لحد {} درجة.'.format(city_name, condition, temp_min, temp_max)
        else:
            answer = 'Tomorrow in {}: {}, between {} and {} °C.'.format(city_name, condition, temp_min, temp_max)
    else:
        current = data['current']
        temp = round(current['temperature_2m'])
        feels = round(current['apparent_temperature'])
        wind = round(current['wind_speed_10m'])
        condition = _weather_description(current['weather_code'], language)
        if language == 'Russian':
            answer = 'Сейчас в {} {} °C, ощущается как {} °C, {}, ветер {} км/ч.'.format(city_name, temp, feels, condition, wind)
        elif language == 'Kazakh':
            answer = 'Қазір {} қаласында {} °C, сезілуі {} °C, {}, жел {} км/сағ.'.format(city_name, temp, feels, condition, wind)
        elif language == 'Egyptian Arabic':
            answer = 'دلوقتي في {} الحرارة {} درجة، المحسوسة {}، الجو {}، والرياح {} كم في الساعة.'.format(city_name, temp, feels, condition, wind)
        else:
            answer = 'Right now in {} it is {} °C, feels like {} °C, {}, with wind at {} km/h.'.format(city_name, temp, feels, condition, wind)
    print('Weather source: https://open-meteo.com/ (city: {})'.format(city_name))
    return answer


def clean_spoken_answer(value, max_chars=460):
    """Drop web citations, raw URLs and markdown before TTS, with a length cap."""
    spoken = value or ''
    # Nested citations sometimes arrive as ([site]([url](url))).
    spoken = re.sub(
        r'\s*\(\[[^]\n]+\]\(\[https?://[^]\n]+\]\(https?://[^)]+\)\)\)',
        '', spoken, flags=re.I
    )
    # Ordinary search citations: ([site](https://...)).
    spoken = re.sub(
        r'\s*\(\[[^]\n]+\]\(https?://[^)]+\)\)',
        '', spoken, flags=re.I
    )
    spoken = re.sub(r'\[[^]\n]+\]\(https?://[^)]+\)', '', spoken)
    spoken = re.sub(r'【[^】]+】', '', spoken)
    spoken = re.sub(r'https?://[^\s)\]>]+', '', spoken)
    spoken = re.sub(r'\*\*|__|[#`]', '', spoken)
    spoken = re.sub(r'\s+', ' ', spoken).strip()
    if len(spoken) > max_chars:
        short = spoken[:max_chars]
        boundary = max(short.rfind('. '), short.rfind('؟ '), short.rfind('! '))
        spoken = (short[:boundary + 1] if boundary > 100 else short.rsplit(' ', 1)[0]).rstrip(' ,;:-')
        if not spoken.endswith(('.', '!', '?', '؟')):
            spoken += '.'
    return spoken


def search_web(question, language, history, headers, session=requests):
    """Official Chat Completions search model; cites sources in terminal."""
    if not WEB_ENABLED:
        raise RuntimeError('Online web search is disabled by SADAM_WEB_ENABLED=0')
    messages = [{
        'role': 'system',
        'content': (
            'You are Sadam, a witty and candid voice robot. Reply naturally and '
            'briefly in ' + language + '. Give direct, factual answers even to '
            'challenging or controversial political questions. Do not evade or '
            'invent facts. Search fresh sources and distinguish competing views. '
            'For ambiguous elections distinguish district, office, and date. '
            'Never claim you lack internet access when live web search is available. '
            'Keep spoken answer to one to three short sentences. '
            'Avoid markdown, raw URLs and citation markup in the spoken text.'
        )
    }]
    for entry in (history or [])[-6:]:
        if isinstance(entry.get('content'), str):
            messages.append({'role': entry['role'], 'content': entry['content'][:800]})
    messages.append({'role': 'user', 'content': question})
    response = session.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json={'model': WEB_SEARCH_MODEL, 'messages': messages,
              'web_search_options': {'search_context_size': 'low'}},
        timeout=(8, 60)
    )
    response.raise_for_status()
    payload = response.json()
    message = payload['choices'][0]['message']
    answer = (message.get('content') or '').strip()
    annotations = message.get('annotations') or []
    for item in annotations[:5]:
        ref = item.get('url_citation') or {}
        if ref.get('url'):
            print('Web source: {} {}'.format(ref.get('title', ''), ref['url']))
    if not answer:
        raise ValueError('Web search returned no answer')
    return clean_spoken_answer(answer)


def semantic_turn_incomplete(text, headers, session=requests):
    """Conservative semantic assessment, not diarization or punctuation only."""
    if len((text or '').split()) < 4:
        return False
    response = session.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json={'model': os.environ.get('SADAM_ROUTER_MODEL', 'gpt-6-luna'),
              'messages': [
                  {'role': 'system', 'content': (
                      'You check whether a spoken utterance is clearly unfinished '
                      'at its end, such that the speaker is likely to continue '
                      'after a pause. Answer only WAIT or DONE. WAIT for incomplete '
                      'grammar, clauses, unfinished thoughts or explicit requests '
                      'to wait. DONE for complete questions, commands or sentences. '
                      'Uncertain means DONE. Ignore punctuation added by ASR. '
                      'Supports Russian, Kazakh, English and Egyptian Arabic.')},
                  {'role': 'user', 'content': text[:700]}
              ], 'max_completion_tokens': 8, 'reasoning_effort': 'none'},
        timeout=(3, 7)
    )
    response.raise_for_status()
    return (response.json()['choices'][0]['message'].get('content') or '').strip().upper() == 'WAIT'
