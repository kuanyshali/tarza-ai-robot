"""Offline coverage for weather, news search and semantic spoken-turn routing."""
import unittest
from types import SimpleNamespace
import sadam_features as feature


class Response:
    def __init__(self, data, status=200):
        self.data = data
        self.status = status
    def json(self):
        return self.data
    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError('HTTP {}'.format(self.status))


class WebWeatherTests(unittest.TestCase):
    def test_weather_route_multilingual(self):
        for phrase in ('Какая погода в Астане?', 'Weather in Astana?',
                       'Ауа райы қандай?', 'الجو عامل إيه في أستانا؟'):
            self.assertTrue(feature.weather_intent(phrase))
            self.assertFalse(feature.online_search_intent(phrase))
        self.assertFalse(feature.weather_intent('What am I holding?'))
        self.assertEqual(feature.weather_city('What is the weather in Dubai today?'), 'Dubai')
        self.assertEqual(feature.weather_city('Какая погода в Астане?'), 'Astana')
        self.assertTrue(feature.wants_tomorrow('weather in Astana tomorrow'))

    def test_live_news_route(self):
        self.assertTrue(feature.online_search_intent('مين فاز في انتخابات ميشيجان امبارح؟'))
        self.assertTrue(feature.online_search_intent('Who won the election yesterday?'))
        self.assertTrue(feature.online_search_intent('Кто президент России?'))
        self.assertFalse(feature.online_search_intent('Tell me a joke'))
        self.assertTrue(feature.online_search_intent('А в Сенате?', recent_context=True))

    def test_weather_uses_real_api_data(self):
        calls = []
        def get(url, **kwargs):
            calls.append((url, kwargs))
            if 'geocoding' in url:
                return Response({'results': [{
                    'name': 'Astana', 'latitude': 51.17, 'longitude': 71.43
                }]})
            return Response({
                'current': {'temperature_2m': -7.4, 'apparent_temperature': -14.6,
                            'wind_speed_10m': 21, 'weather_code': 71},
                'daily': {'temperature_2m_min': [-9, -11],
                          'temperature_2m_max': [-5, -4],
                          'weather_code': [71, 3]}
            })
        fake = SimpleNamespace(get=get)
        answer = feature.get_weather('Какая погода в Астане?', 'Russian', session=fake)
        self.assertIn('-7', answer)
        self.assertIn('-15', answer)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1]['params']['timezone'], 'auto')
        tomorrow = feature.get_weather('Weather in Astana tomorrow', 'English', session=fake)
        self.assertIn('Tomorrow', tomorrow)
        self.assertIn('-11', tomorrow)

    def test_weather_no_data_fails_without_hallucinating(self):
        def get(url, **kwargs):
            return Response({'results': []})
        with self.assertRaises(ValueError):
            feature.get_weather('weather in Unknowncity', session=SimpleNamespace(get=get))

    def test_web_search_calls_live_search_model_and_prints_sources(self):
        observed = []
        def post(url, **kwargs):
            observed.append((url, kwargs))
            return Response({'choices': [{'message': {
                'content': 'The latest result is confirmed.【1】',
                'annotations': [{'url_citation': {
                    'url': 'https://example.org/results', 'title': 'Results'
                }}]
            }}]})
        answer = feature.search_web('Who won yesterday?', 'English',
                                    [{'role':'user','content':'Michigan Senate'}],
                                    {'Authorization':'Bearer test'},
                                    session=SimpleNamespace(post=post))
        self.assertEqual(answer, 'The latest result is confirmed.')
        self.assertIn('web_search_options', observed[0][1]['json'])
        self.assertEqual(observed[0][1]['json']['model'], feature.WEB_SEARCH_MODEL)

    def test_semantic_end_only_waits_on_explicit_wait(self):
        observed = []
        def post(url, **kwargs):
            observed.append(kwargs)
            return Response({'choices': [{'message': {'content': 'WAIT'}}]})
        wait = feature.semantic_turn_incomplete(
            'I wanted to tell you something and',
            {'Authorization':'Bearer test'}, session=SimpleNamespace(post=post))
        self.assertTrue(wait)
        self.assertEqual(observed[0]['json']['messages'][0]['role'], 'system')
        self.assertFalse(feature.semantic_turn_incomplete(
            'Hi', {}, session=SimpleNamespace(post=post)))


if __name__ == '__main__':
    unittest.main()
