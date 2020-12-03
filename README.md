# TakeHomeExam

You are doing ML/DL analysis for trading cryptocurrency to make a profit. Our target
cryptocurrency is popular Bitcoin (BTC).

* 2018-07-trade.csv data format: This file contains the transaction history records of BTC-KRW
  market during July 1-2, 2018. It’s a basically a file showing the sequence of Buy and Sell
  transactions for BTC. And it may be possible to use this data as training/test dataset for smart
  transaction agent for cryptocurrency market.

- timestamp: time of your order in place, yyyy-mm-dd HH:MM:SS
- quantity: BTC size in transaction
- price: 1 BTC price in Korean Won (KRW)
- fee: commission, you can ignore this
- amount: the total amount of KRW in transaction, quantity x price
- side: 0 a Buy (Bid), 1 a Sell (Ask)
- The following example is May, 2018.
…
2018-05-01 01:06:24,0.3382,10162000,0,3436788,1
…
At May-1-2018 01:06:24, I sold 0.3382 BTC for the price of 10,162,000 KRW.
- 2018-07-01-orderbook.csv and 2018-07-02-orderbook.csv data format: This file contains socalled
orderbook (market data) of BTC-KRW market. The data has the order records of
willingness to Buy and Sell BTC for every second. Every second data have top 15 levels of Buy
and Sell (the first 15 lines are representing Buy requests, the next 15 lines are representing Sell
requests, so the total of 30 lines are recorded for every second.)
- price: 1 BTC price in KRW
- quantity: BTC size in market
- type: 0 a Buy (Bid), 1 a Sell (Ask)
- timestamp: time of market, yyyy-mm-dd HH:MM:SS.us



## Task 1

#### Compute the total profit of July 1-2 in KRW. 

#### It simply means how much money do we make or lose? 

#### The profit calculation moment should be when the accumulative quantity is 0 (only consider 4 digit floating number, ignore the rest).

```python
import pandas as pd
trade = pd.read_csv('../Data/2018-07-trade.csv')

# quantity 값이 sell, buy 상관없이 양수이기 때문에 side 값을 기준으로 분류하여 총합을 구해 비교하였다.
# 모든 거래가 끝났을때 sell과 buy의 양이 같음을 알 수 있다.
print(trade['quantity'].groupby(trade['side']).sum())
# side
# 0    83.912906
# 1    83.912940

# 모든 거래가 끝났을때 코인이 남지 않기때문에 판매금액과 구매금액을 비교하면 최종 수익을 알 수 있다.
# amount값은 sell은 양수 buy는 음수이기 때문에 모든 거래 금액의 총합을 구하면 최종 수익을 확인 할 수 있다.
print(trade['amount'].groupby(trade['side']).sum())
# side
# 0   -599313768
# 1    599896414

print(trade['amount'].sum())
# 582646
```



## Task 2

#### Report how many Buy and Sell transactions separately. 

#### Draw a time-series bar graph illustrating changes in transaction counts 

#### (x-axis: 10 minutes interval, y1-axis: Sell, y2-axis: Buy)



## Task 3

#### For this task, you will need orderbook.csv files to work with. Compute the following

#### features and modify 2018-07-trade.csv. Submit your new csv file: 2018-07-trade-new.csv. For

#### your final csv file, you will remove a few existing columns and add three new columns: 

#### midprice, bookfeature, and alpha. 

#### In order to compute these two, check out the following:

* How to compute MidPrice
  MidPrice = (top_ask_price + top_bid_price) / 2 //ask means Sell, bid means Buy.

* How to compute BookFeature 

  askQty = quant_orderbook_ask.values.avg()  //average quantity of all levels for Sell
  bidQty = quant_orderbook_bid.values.avg() //likewise for Buy
  askPx = price_orderbook_ask.values.avg() //average price of all levels for Sell
  bidPx = price_orderbook_bid.values.avg() //likewise for Buy
  book_price = (((askQty*bidPx)/bidQty) + ((bidQty*askPx)/askQty)) / (bidQty+askQty)
  BookFeature = (book_price - mid_price)

* How to compute Alpha:
  Alpha = 0.002 * BookFeature * MidPrice



## Task 4

#### Explain your plan how you use the data file from Task 3 to create the smart trading agent. 

#### You can explain how you create training and test dataset. 

#### And show how to use ML or Neural Network (using Keras, perhaps) to create the learning agent for cryptocurrency transaction. 

### 