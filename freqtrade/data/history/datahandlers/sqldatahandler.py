import logging

from pandas import DataFrame, read_feather, to_datetime

from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType, TradingMode

from .idatahandler import IDataHandler

from sqlalchemy import create_engine
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SqlDataHandler(IDataHandler):

    def __init__(self, datadir: Path):
        super().__init__(datadir)
        self.engine = create_engine("")

    @classmethod
    def _get_file_extension(cls) -> str:
        """
        just return sql string
        :return:
        """
        return "sql"

    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        """
        no need to store newest data, data warehouse already did
        :param pair:
        :param timeframe:
        :param data:
        :param candle_type:
        :return:
        """
        pass

    def _ohlcv_load(self, pair: str, timeframe: str, timerange: TimeRange | None, candle_type: CandleType) -> DataFrame:
        coin_symbol = pair.split("/")[0].lower()
        start_time_str = timerange.start_fmt
        end_time_str = timerange.stop_fmt
        sql = f"select open_timestamp as date,open,high,low,close,volume from dwd_{coin_symbol}_kline_{timeframe} where open_time between '{start_time_str}' and '{end_time_str}' order by open_time"
        logger.info("using sql "+sql+" querying data")
        df = pd.read_sql(sql, self.engine)
        logger.info("querying data length: "+str(len(df)))
        df = df.astype({
            'open': 'float',
            'high': 'float',
            'low': 'float',
            'close': 'float',
            'volume': 'float'
        })
        df['date']=pd.to_datetime(df['date'],unit='ms',utc=True)
        return df

    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: DataFrame, trading_mode: TradingMode) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        :param trading_mode: Trading mode to use (used to determine the filename)
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        self.create_dir_if_needed(filename)
        data.reset_index(drop=True).to_feather(filename, compression_level=9, compression="lz4")

    def trades_append(self, pair: str, data: DataFrame):
        """
        no need to implement
        :param pair: Pair - used for filename
        :param data: Dataframe containing trades
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(self, pair: str, trading_mode: TradingMode, timerange: TimeRange | None = None) -> DataFrame:
        """
        Load a pair from file, either .json.gz or .json
        # TODO: respect timerange ...
        :param pair: Load trades for this pair
        :param trading_mode: Trading mode to use (used to determine the filename)
        :param timerange: Timerange to load trades for - currently not implemented
        :return: Dataframe containing trades
        """
        filename = self._pair_trades_filename(self._datadir, pair, trading_mode)
        if not filename.exists():
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)

        tradesdata = read_feather(filename)

        return tradesdata
