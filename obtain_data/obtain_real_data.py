#
import time
import pymongo
import math
import pandas as pd


class GenerateRawData(object):
    def __init__(self):
        self.client = pymongo.MongoClient(host='localhost', port=37018)  # mongodb默认端口是27017
        self.db = self.client.trajectory  # 指定数据库, 如果没有则会自动创建


    def change_geo(self,geo1):
        du = math.floor(geo1 / 10000)
        fen = math.floor((geo1 - du * 10000) / 100) / 60
        miao = (geo1 - du * 10000 - fen * 60 * 100) / 3600
        geo_c = du + fen + miao
        # geo_cc=geo/10000
        return geo_c

    def get_plan_data(self, date):
        collection = self.db[date]
        filter_db = {}
        output = {'_id': 0, 'flight.fmeid': 1, 'departure.departureAerodrome': 1, 'departure.actualTakeoffTime': 1, 'arrival.arrivalAerodrome': 1, 'arrival.acturalLandingTime': 1,
                  'trackTratectory': 1}
        result = collection.find(filter_db, output)
        print("There are {0} flights of date: {1}".format(str(result.count()), date))

        all_flights = []
        process_count = 0

        for r in result:
            process_count += 1
            if process_count % 10000 == 0:
                print("Processing Count: {0}".format(process_count))
            is_miss_info = False
            # print(r)
            #_id = r['_id']
            flight = r['flight']
            _id = flight['fmeid']
            departure_airport = r.get('departure', {}).get('departureAerodrome', None)
            departure_time = r.get('departure', {}).get('actualTakeoffTime', None)
            arrival_airport = r.get('arrival', {}).get('arrivalAerodrome', None)
            arrival_time = r.get('arrival', {}).get('acturalLandingTime', None)
            if departure_time is not None:
                departure_time = time.mktime(time.strptime(departure_time[:12], "%Y%m%d%H%M"))
            if arrival_time is not None:
                arrival_time = time.mktime(time.strptime(arrival_time[:12], "%Y%m%d%H%M"))
            if departure_airport is None or departure_time is None or arrival_airport is None or arrival_time is None:
                is_miss_info = True
            flight_routes = []
            for fr in r.get('trackTratectory', []):
                trackTime, trackPosition = fr.get('trackTime', None), fr.get('trackPosition', None)
                if trackTime is not None and trackTime[:12] != '000000000000':
                    trackTime = time.mktime(time.strptime(trackTime[:12], "%Y%m%d%H%M"))
                else:
                    trackTime = None
                if trackPosition is not None:
                    long, lat = trackPosition.split(',')
                    long, lat = self.change_geo(int(long)), self.change_geo(int(lat))
                else:
                    long, lat = None, None
                fr_info = (trackTime, long, lat)
                flight_routes.append(fr_info)
                if None in fr_info:
                    is_miss_info = True
            flight_routes = sorted(flight_routes, key=lambda fr: fr[0])
            all_flights.append((_id, departure_airport, departure_time, arrival_airport, arrival_time, flight_routes, is_miss_info))

        df_plan = pd.DataFrame(all_flights,
                               columns=['fmeid', 'departure_airport', 'departure_time', 'arrival_airport', 'arrival_time',
                                        'flight_routes', 'is_miss_info'])
        return df_plan


if __name__ == "__main__":
    test = GenerateRawData()
    date = '202106'
    df_plan = test.get_plan_data(date)
    df_plan.to_csv("./FlightsReal_{}.csv".format(date))
