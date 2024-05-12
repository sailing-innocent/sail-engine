import datetime 

if __name__ == "__main__":
    now = datetime.date.today()
    tencent_time = datetime.date(2024,6,24)
    UGDAP_time = datetime.date(2024,5,8)
    SIG_time = datetime.date(2024,5,12)
    NIPS_time = datetime.date(2024,5,22)
    handred_days_milestone = datetime.date(2024,6,2)
    d_hundred_days_milestone = datetime.date(2024,9,10)

    print("Today is: ", now)
    print("NIPS: ", NIPS_time - now)
    print("100 days milestone: ", handred_days_milestone - now)
    print("Tencent UGDAP: ", tencent_time - now)
    print("200 days milestone: ", d_hundred_days_milestone - now)
