name := "DataMart"

version := "0.1"

scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "datamart"
  )

val sparkVersion = "3.5.1"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.datastax.spark" %% "spark-cassandra-connector" % "3.5.0",
)