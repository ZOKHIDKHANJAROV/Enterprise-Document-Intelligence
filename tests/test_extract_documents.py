from text_quality import fix_mojibake, noise_ratio, text_quality


def test_fix_mojibake_repairs_cp1251_read_as_cp1252():
    broken = 'ÏËÀÑÒÈÍÛ ÒÂÅÐÄÎÑÏËÀÂÍÛÅ ' * 5
    assert fix_mojibake(broken).startswith('ПЛАСТИНЫ ТВЕРДОСПЛАВНЫЕ')
    assert fix_mojibake('Обычный текст приказа') == 'Обычный текст приказа'
    assert fix_mojibake('Café crème') == 'Café crème'


def test_text_quality_separates_readable_from_broken():
    readable = 'Ответственным за исполнение приказа назначить начальника отдела кадров. ' * 10
    assert text_quality(readable) > 0.8
    assert text_quality('/G01/G02/G03/G04/G05 ' * 50) < 0.1
    assert text_quality("l;1l,tl l] l l 1l lI ))\nl l)\"])i)l)ll |l;l)l)ll]j1l l " * 20) < 0.3
    assert text_quality('\x05\x14\x05\x15\x16\n\x0e' * 30) == 0.0
    assert text_quality('') == 0.0


def test_noise_ratio_flags_ocr_noise_not_lists_or_numbers():
    assert noise_ratio('Мехжш,t,нн ,ttухофilЗt} ýýjIHш хспраёнила и{100нки}лг') > 0.5
    assert noise_ratio('ГОСТ 12.1.004–91, ГОСТ 12.1.005–88, ГОСТ 12.1.007–76') == 0.0
    assert noise_ratio('Ответственным назначен начальник отдела кадров.') == 0.0
    assert noise_ratio('Oltin saralash fabrikasi probir analitik nazorat') == 0.0
    assert noise_ratio('01.07.81') == 0.0
    assert noise_ratio('') == 0.0
