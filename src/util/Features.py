from enum import Enum, unique


@unique
class Invested(Enum):
    LONG = 'LONG'
    SHORT = 'SHORT'


@unique
class Features(Enum): pass


# in case we want to add functionality later


class SnpFeatures(Features):
    ASK = 'ASK'
    BID = 'BID'
    EARN_FOR_COMMON = 'EARN_FOR_COMMON'
    EBITDA = 'EBITDA'
    FUND_NET_ASSET_VAL = 'FUND_NET_ASSET_VAL'
    HIGH = 'HIGH'
    LAST_PRICE = 'LAST_PRICE'
    LOW = 'LOW'
    OPEN = 'OPEN'
    PE_RATIO = 'PE_RATIO'
    SHORT_AND_LONG_TERM_DEBT = 'SHORT_AND_LONG_TERM_DEBT'
    TOTAL_ASSETS = 'TOTAL_ASSETS'
    TOTAL_EQUITY = 'TOTAL_EQUITY'
    TOT_BUY_REC = 'TOT_BUY_REC'
    TOT_HOLD_REC = 'TOT_HOLD_REC'
    TOT_SELL_REC = 'TOT_SELL_REC'


class EtfFeatures(Features):
    ASK = 'ASK'
    BID = 'BID'
    EARN_FOR_COMMON = 'EARN_FOR_COMMON'
    EBITDA = 'EBITDA'
    FUND_NET_ASSET_VAL = 'FUND_NET_ASSET_VAL'
    HIGH = 'HIGH'
    LAST_PRICE = 'LAST_PRICE'
    LOW = 'LOW'
    OPEN = 'OPEN'
    PE_RATIO = 'PE_RATIO'
    SHORT_AND_LONG_TERM_DEBT = 'SHORT_AND_LONG_TERM_DEBT'
    TOTAL_ASSETS = 'TOTAL_ASSETS'
    TOTAL_EQUITY = 'TOTAL_EQUITY'
    TOT_BUY_REC = 'TOT_BUY_REC'
    TOT_HOLD_REC = 'TOT_HOLD_REC'
    TOT_SELL_REC = 'TOT_SELL_REC'
