import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Jiang20011010...",
  database="mysql"
)

print(mydb)