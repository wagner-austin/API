dsenames(permno,namedt,nameendt,shrcd,exchcd) AS (
  VALUES (1,DATE '2015-01-01',DATE '2024-12-31',10,1),
         (2,DATE '2015-01-01',DATE '2024-12-31',10,1)
),
stkdelists(permno,delistingdt,deldtprc,deldtprcflg,delactiontype,
delstatustype,delreasontype,delpaymenttype,delpermno,delpermco,
delret,delretmisstype,delnextdt,delnextprc,delnextprcflg,delamtdt,
deldivamt,deldistype,deldlydt) AS (
  SELECT p,event,10.0::numeric,'BA'::varchar,NULL::varchar,
    NULL::varchar,NULL::varchar,NULL::varchar,0,0,
    0.0::numeric,'MV'::varchar,NULL::date,NULL::numeric,NULL::varchar,valuation,
    0.0::numeric,NULL::varchar,event
  FROM (VALUES
    (1,DATE '2019-06-03',NULL::date),
    (1,DATE '2019-12-31',DATE '2020-01-10'),
    (2,DATE '2019-06-03',NULL::date),
    (1,DATE '2020-01-01',NULL::date),
    (1,DATE '1990-01-01',NULL::date)
  ) AS evidence(p,event,valuation)
),
stkdistributions(permno,disexdt,disseqnbr,disordinaryflg,distype,
disfreqtype,dispaymenttype,disdetailtype,distaxtype,disorigcurtype,
disdivamt,disfacpr,disfacshr,disdeclaredt,disrecorddt,dispaydt,
dispermno,dispermco,disamountsourcetype) AS (
  SELECT p,event,1,'N'::varchar,'CD'::varchar,NULL::varchar,
    'CASH'::varchar,NULL::varchar,NULL::varchar,NULL::varchar,
    0.0::numeric,-1.0::numeric,0.0::numeric,NULL::date,NULL::date,payment,
    2,3,'NA'::varchar
  FROM (VALUES
    (1,DATE '2019-06-03',DATE '2019-06-07'),
    (1,DATE '2019-12-31',DATE '2020-01-10'),
    (2,DATE '2019-06-03',NULL::date),
    (1,DATE '2020-01-01',NULL::date),
    (1,DATE '1990-01-01',NULL::date)
  ) AS evidence(p,event,payment)
)
