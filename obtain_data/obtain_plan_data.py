#
import pymongo
import pandas as pd
import time
import math


class GenerateRawRealData(object):
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
        output = {'_id': 0, 'flight.fmeid': 1, 'departure.departureAerodrome': 1, 'departure.estimateOffBlockTime': 1, 'arrival.arrivalAerodrome': 1, 'arrival.estimatedInBlockTime': 1,
                  'flightRouteResult.currentFlightRoutes.flightRouteElements': 1}
        result = collection.find(filter_db, output)
        print("There are {0} flights of date: {1}".format(str(result.count()), date))

        all_flights = []
        process_count = 0
        process_error = 0

        for r in result:
            process_count += 1
            if process_count % 10000 == 0:
                print("Processing Count: {0}".format(process_count))
            is_miss_info = False
            flight = r['flight']
            _id = flight['fmeid']
            departure_airport = r.get('departure', {}).get('departureAerodrome', None)
            departure_time = r.get('departure', {}).get('estimateOffBlockTime', None)
            arrival_airport = r.get('arrival', {}).get('arrivalAerodrome', None)
            arrival_time = r.get('arrival', {}).get('estimatedInBlockTime', None)
            if departure_time is not None:
                departure_time = time.mktime(time.strptime(departure_time[:12], "%Y%m%d%H%M"))
            if arrival_time is not None:
                arrival_time = time.mktime(time.strptime(arrival_time[:12], "%Y%m%d%H%M"))
            if departure_airport is None or departure_time is None or arrival_airport is None or arrival_time is None:
                is_miss_info = True
            flight_routes = []
            for fr in r.get('flightRouteResult', {}).get('currentFlightRoutes', {}).get('flightRouteElements', []):
                trackTime, lat, long = fr.get('pasttime', None), fr.get('latitude', None), fr.get('longitude', None)
                if trackTime is not None and trackTime[:12] != '000000000000':
                    trackTime = time.mktime(time.strptime(trackTime[:12], "%Y%m%d%H%M"))
                else:
                    trackTime = None
                if long is not None:
                    long = self.change_geo(int(long))
                else:
                    long = None
                if lat is not None:
                    lat = self.change_geo(int(lat))
                else:
                    lat = None
                #
                fr_info = (trackTime, long, lat)
                if None not in fr_info:
                    flight_routes.append(fr_info)
                else:
                    is_miss_info = True
            try:
                flight_routes = sorted(flight_routes, key=lambda fr: int(fr[0]))
            except:
                process_error += 1
                if process_error % 10 == 0:
                    print("error flights:", process_error)
                continue

            all_flights.append((_id, departure_airport, departure_time, arrival_airport, arrival_time, flight_routes, is_miss_info))

        df_plan = pd.DataFrame(all_flights,
                               columns=['_id', 'departure_airport', 'departure_time', 'arrival_airport', 'arrival_time',
                                        'flight_routes', 'is_miss_info'])
        return df_plan

    def load_airport_info(self, name):
        db_ap = self.client.airspacedata
        collection = db_ap[name]
        filter_db = {}
        output = {'APNAME': 1, 'CITY': 1, 'IDENTIFIER': 1, 'LONGITUDE': 1, 'LATITUDE': 1}
        result = collection.find(filter_db, output)
        airport_info = []
        for r in result:
            airport_info.append((r['APNAME'], r['CITY'], r['IDENTIFIER'], r['LONGITUDE'], r['LATITUDE']))
        df_airport = pd.DataFrame(airport_info, columns=['APNAME', 'CITY', 'IDENTIFIER', 'LONGITUDE', 'LATITUDE'])
        return df_airport


if __name__ == "__main__":
    test = GenerateRawRealData()
    date = '202106'
    df_plan = test.get_plan_data(date)
    df_plan.to_csv("FlightsPlan_{}.csv".format(date))
    # df_airport = test.load_airport_info('airport')
    # df_airport.to_csv('./AirportInfo.csv')