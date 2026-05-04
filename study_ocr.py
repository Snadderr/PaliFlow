"""Curated OCR strings for the 60 study images."""

IMAGE_OCR: dict[int, str] = {
    # ─── Set 1 ───
    2: "USA TODAY (Media watermark); AP (Media watermark)",
    4: "18+; БАТАЛЬЙОН К-2 (K-2 Battalion); 54 ОМБР (54th Separate Mechanized Brigade)",
    44: (
        "18+; СПІЛЬНОТА СТЕРНЕНКА (Sternenko Community); "
        "БАТАЛЬЙОН К-2 (K-2 Battalion); "
        "54 ОМБР (54th Separate Mechanized Brigade)"
    ),
    50: (
        "TEW22 (Media watermark); "
        "СВОБОДА ІДУ НА ВИ (Freedom, I come at you); "
        "SUBSCRIBE (Media ticker); "
        "FOLLOW US ON facebook (Social media handle)"
    ),
    57: (
        "18+; "
        'T-72 "КОНІВНИК" (T-72 "Groom/Horseman"); '
        'T-72 "СОКІЛ" (T-72 "Falcon"); '
        "БАТАЛЬЙОН К-2 (K-2 Battalion); "
        "54 ОМБР (54th Separate Mechanized Brigade); "
        "ЧАМО-82 (CHAMO-82)"
    ),
    60: (
        "103rd Territorial Brigade (Unit designation); "
        "The Sun (Media watermark); "
        "СПІЛЬНОТА СТЕРНЕНКА (Sternenko Community); "
        "The Kamikaze drones successfully destroy enemy armored and cargo vehicles "
        "and electronic warfare equipment by crashing into them at speed. (News ticker); "
        "RUSORIZ_6548 (Unit designation/Handle)"
    ),
    79: "БАТАЛЬЙОН К-2 (K-2 Battalion); 54 ОМБР (54th Separate Mechanized Brigade)",
    92: None,
    128: None,
    146: None,
    150: "ВОЕННАЯ ХРОНИКА (Military Chronicle - Media watermark)",
    158: None,
    201: None,
    236: "TECH ULTIMATE (Media watermark)",
    240: None,
    244: None,
    248: None,
    7: None,
    26: None,
    80: None,

    # ─── Set 2 ───
    0: "18+; БАТАЛЬЙОН К-2 (K-2 Battalion); 54 ОМБР (54th Separate Mechanized Brigade)",
    6: (
        "Newsflash (Media watermark); "
        "БМД 4 (BMD-4 - Unit/Vehicle designation); "
        "3x (Optical zoom indicator); "
        "-90° (Angle indicator); "
        "AF (Auto-focus indicator); "
        "The Sun (Media watermark)"
    ),
    23: (
        "Newsflash (Media watermark); "
        "БМП-1 (BMP-1 - Unit/Vehicle designation); "
        "The Sun (Media watermark)"
    ),
    55: "13 kanal (Media watermark)",
    64: (
        "ЕПІЧНИЙ РОЗГРОМ РОСІЯН НА ДОНЕЧЧИНІ "
        "(Epic rout of Russians in the Donetsk region - News ticker); "
        "не врятувала російських танкістів... "
        "(did not save Russian tank crews... - News ticker)"
    ),
    70: None,
    71: None,
    81: "18+; БАТАЛЬЙОН К-2 (K-2 Battalion); 54 ОМБР (54th Separate Mechanized Brigade)",
    83: None,
    123: "54500 (Vehicle registration/identification); 40 (Tactical marking/identifier)",
    149: "БАТАЛЬЙОН К-2 (K-2 Battalion); 54 ОМБР (54th Separate Mechanized Brigade)",
    163: None,
    165: (
        "VOG-25 RUSSIAN (Unit/Weaponry designation); "
        "1,0x (Optical zoom indicator); "
        "AF (Auto-focus indicator)"
    ),
    166: (
        "t.me/escadrone (Social media handle); "
        "Спонсор ураження: Спільнота С.Стерненка "
        "(Sponsor of the strike: S. Sternenko Community - News ticker)"
    ),
    190: None,
    211: "LOW BATTERY (Status indicator)",
    217: None,
    224: None,
    227: None,
    237: "БАРКАС (Barkas - Unit/Textual Identifier)",

    # ─── Set 3 ───
    10: "18+; БАТАЛЬЙОН К-2 (K-2 Battalion); 54 ОМБР (54th Separate Mechanized Brigade)",
    17: None,
    24: "18+; БАТАЛЬЙОН К-2 (K-2 Battalion); 54 ОМБР (54th Separate Mechanized Brigade)",
    38: None,
    59: (
        "m (Media watermark/Logo); "
        "Some bombs miss their targets by a few feet but others score a direct hit, "
        "setting fire to one tank. (News ticker)"
    ),
    72: None,
    84: (
        "18+; "
        'T-72 "КОНІВНИК" (T-72 "Groom/Horseman"); '
        'T-72 "СОКІЛ" (T-72 "Falcon"); '
        "БАТАЛЬЙОН К-2 (K-2 Battalion); "
        "54 ОМБР (54th Separate Mechanized Brigade)"
    ),
    95: "18+; БАТАЛЬЙОН К-2 (K-2 Battalion); 54 ОМБР (54th Separate Mechanized Brigade)",
    113: None,
    145: "Image © 2013 DigitalGlobe (Media watermark)",
    153: (
        "ВОЕННАЯ ХРОНИКА (Military Chronicle - Media watermark); "
        "@MILCHRONICLES (Social media handle)"
    ),
    156: None,
    161: None,
    162: None,
    171: "AIR (HUD indicator)",
    174: None,
    181: "БАРКАС (Barkas - Unit designation)",
    182: "РОНІНИ (Ronins - Unit designation); LXV (Unit designation/65th)",
    232: None,
    242: None,
}


def get_ocr(img_idx: int) -> str | None:
    return IMAGE_OCR.get(img_idx)
