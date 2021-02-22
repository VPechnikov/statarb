from typing import List
from src.Cointegrator import CointegratedPair
from src.Portfolio import Portfolio, Position
from src.util.Features import PositionType
from datetime import date, timedelta
from src.Filters import Filters


class Decision:
    def __init__(self, position: Position,
                 old_action: PositionType,
                 new_action: PositionType):
        self.position: Position = position
        self.new_action: PositionType = new_action
        self.old_action: PositionType = old_action


class SignalGenerator:

    def __init__(self,
                 port: Portfolio,
                 entry_z: float,
                 exit_z: float,
                 emergency_delta_z: float):
        self.port: Portfolio = port
        self.entry_z: float = entry_z
        self.exit_z: float = exit_z
        self.emergency_delta_z: float = emergency_delta_z
        self.time_stop_loss = 30
        self.open_count = 0
        self.natural_close_count = 0
        self.emergency_close_count = 0  # The number of previously cointegrated and traded pairs that are no longer cointegrated.
        self.time_stop_loss_count = 0
        self.filter = Filters()
        self.volumn_shock_filter = 0


    def make_decision(self, pairs: List[CointegratedPair]) -> List[Decision]:
        """
        The function:
        -Creates 2 lists: one with the tickers of the cointegrated pairs and one with the tickers of current position pairs
        -Iterates through cointegrated pairs and checks if there are pairs which are closed and must be open(conditions
        about reaching the threshold point and possible existence of volume shock must hold). If so, open positions
        -Iterates through current positions and checks if there are pairs which are no longer cointegrated. In this case,
        we close these positions
        -Closes the positions which are profitable after the time limit
        -Closes the positions which have reached either the emergency point(loss) or the exit point(profit) at any time
        -Keeps open the remaining positions
        -Returns a list with decisions for each pair(contains old action and new action on the pair)
        """

        positions = self.port.cur_positions  # the port.cur_positions has all the information of the pairs that are currnetly traded/open (tickers, weights, quantities,pnl, etc..
        current_posn_pairs = [(i.asset1, i.asset2) for i in positions]  # return a list with the tickers of traded pairs
        coint_pairs = [i.pair for i in pairs]  # a list with tickers of cointegrated pairs
        today = self.port.current_window.window_end
        decisions = []
        for coint_pair in pairs:
            # if coint_pair not invested, check if we need to open position
            if coint_pair.pair not in current_posn_pairs:
                if coint_pair.recent_dev_scaled > self.entry_z:
                    # l = long pair = long x short y
                    p1, p2 = coint_pair.pair
                    shock = self.filter.run_volume_shock_filter_single_pair(coint_pair.pair,
                                                                            self.port.current_window)  # If both etf and stock have shocks or none of them has shock,
                    # shock is False, otherwise the shock = True meaning that only one of the two assets has a shock, so we do not continue with this trade.
                    if not shock:  # if shock is an empty list then the condition (not shock) is True
                        decisions.append(
                            Decision(
                                position=Position(
                                    ticker1=p1,
                                    ticker2=p2,
                                    weight1=coint_pair.scaled_beta,
                                    weight2=1 - coint_pair.scaled_beta,
                                    investment_type=PositionType.LONG,
                                    init_z=coint_pair.recent_dev_scaled,
                                    init_date=today),
                                old_action=PositionType.NOT_INVESTED,
                                new_action=PositionType.LONG,
                            )
                        )
                        self.open_count += 1
                        #print("Long")

                    else:
                        self.volumn_shock_filter += 1  # It counts the cointegrated pairs with only one of the two assets having volume shock

                elif coint_pair.recent_dev_scaled < - self.entry_z:
                    # s = short pair = long y short x
                    p1, p2 = coint_pair.pair
                    shock = self.filter.run_volume_shock_filter_single_pair(coint_pair.pair,
                                                                            self.port.current_window)
                    if not shock:

                        decisions.append(
                            Decision(
                                position=Position(
                                    ticker1=p1,
                                    ticker2=p2,
                                    weight1=-coint_pair.scaled_beta,
                                    weight2=coint_pair.scaled_beta + 1,
                                    investment_type=PositionType.SHORT,
                                    init_z=coint_pair.recent_dev_scaled,
                                    init_date=today),
                                old_action=PositionType.NOT_INVESTED,
                                new_action=PositionType.SHORT,

                            )
                        )
                        self.open_count += 1
                        #print("short")
                    else:
                        self.volumn_shock_filter += 1

        # loop through all invested position
        for position in positions:
            position_pair = (position.asset1, position.asset2)  # tickers
            # if pair not cointegrated, exit position
            if position_pair not in coint_pairs:  # we check whether the pairs currently traded are still cointegrated. If they are not, we close the position.
                decisions.append(
                    Decision(
                        position=position,
                        old_action=position.position_type,
                        new_action=PositionType.NOT_INVESTED))
                self.emergency_close_count += 1  # The number of previously cointegrated and traded pairs that are no longer cointegrated.

            else:  # they are still cointegrated
                idx = coint_pairs.index(position_pair)
                coint_pair = pairs[idx]
                # if position passed time limit and recent_dev has not reached exit_z, but the trade is stil profitable close the trade.
                # In case we pass the time limit but we have losses because we are between emergency_delta_z and entry z in absolute values we don't close the trade.
                if today > (position.init_date + timedelta(self.time_stop_loss)) and \
                        (abs(coint_pair.recent_dev_scaled) < self.entry_z):
                    decisions.append(
                        Decision(
                            position=position,
                            old_action=position.position_type,
                            new_action=PositionType.NOT_INVESTED))
                    self.time_stop_loss_count += 1  # counts the number of trades closed because we reached the time limit.
                # else, check if need to exit
                else:
                    if position.position_type is PositionType.LONG:
                        natural_close_required = coint_pair.recent_dev_scaled < self.exit_z  # close with profit
                        emergency_close_required = coint_pair.recent_dev_scaled > \
                                                   (self.emergency_delta_z + position.init_z)  # close with loss !!!!

                        if natural_close_required or emergency_close_required:
                            decisions.append(
                                Decision(
                                    position=position,
                                    old_action=PositionType.LONG,
                                    new_action=PositionType.NOT_INVESTED)
                            )
                            if natural_close_required:
                                self.natural_close_count += 1
                            else:
                                self.emergency_close_count += 1  # closed trades with loss due to z-score above 3 (stop loss count)
                        else:
                            # no need to close, so keep the position open
                            decisions.append(
                                Decision(
                                    position=position,
                                    old_action=PositionType.LONG,
                                    new_action=PositionType.LONG)
                            )

                    elif position.position_type is PositionType.SHORT:

                        natural_close_required = coint_pair.recent_dev_scaled > -self.exit_z  # profit
                        emergency_close_required = coint_pair.recent_dev_scaled < \
                                                   (
                                                               position.init_z - self.emergency_delta_z)  # stop loss !!!!!!!!!!

                        if natural_close_required or emergency_close_required:
                            decisions.append(
                                Decision(
                                    position=position,
                                    old_action=PositionType.SHORT,
                                    new_action=PositionType.NOT_INVESTED)
                            )
                            if natural_close_required:
                                self.natural_close_count += 1
                            else:
                                self.emergency_close_count += 1
                        else:
                            # no need to close, so keep the position open
                            decisions.append(
                                Decision(
                                    position=position,
                                    old_action=PositionType.SHORT,
                                    new_action=PositionType.SHORT)
                            )
        print("open count: ", self.open_count)
        print("natural close count: ", self.natural_close_count)
        print("emergency close count: ", self.emergency_close_count)
        print("time stop-loss close count: ", self.time_stop_loss_count)
        return decisions
