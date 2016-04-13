import calendar
#set firstweekday=0
cal= calendar.Calendar(firstweekday=0)
# for x in cal.iterweekdays():
#     print(x)

data_week = []
#for year in range(2004,2009):
#    for month in range(1,12):
# year = 2004
# month = 1
# for week in cal.monthdayscalendar(2004, 1):
#     for (weekday,monthday) in enumerate(week):
#         if monthday == 0:
#             continue

#print cal.iterweekdays(2004)

from dateutil import rrule
from datetime import datetime, timedelta

now = datetime.now()
hundredDaysLater = now + timedelta(days=100)

start = datetime(2004,1,5)
end   = datetime(2004,3,5)
for dt in rrule.rrule(rrule.WEEKLY, dtstart=start, until=end):
    print dt, dt.weekday()
    for dt2 in rrule.rrule(rrule.DAILY, dtstart=dt, until=dt+timedelta(days=6)):
        print "    ", dt2, dt2.weekday()



#print calendar.weekday(2004, 1, 5)