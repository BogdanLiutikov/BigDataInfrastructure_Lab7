-- Create a keyspace
CREATE KEYSPACE IF NOT EXISTS food WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : '1' };

-- Create a table
CREATE TABLE IF NOT EXISTS food.data (
id int PRIMARY KEY,
features text
);

CREATE TABLE IF NOT EXISTS food.predictions (
id int PRIMARY KEY,
features text,
prediction int,
timestamp timestamp
);
