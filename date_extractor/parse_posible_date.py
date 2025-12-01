from dateutil import parser

def parse_possible_date(s):
    try:
        return parser.parse(s, dayfirst=True, fuzzy=True)
    except:
        return None