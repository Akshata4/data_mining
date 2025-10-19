# Apachebeam Notebook
## Video Demo
[Watch the demo on SJSU OneDrive](https://sjsu0-my.sharepoint.com/:v:/g/personal/akshata_madavi_sjsu_edu/EY6oAUcu7fhJhHWsHGvboEIBErALXtrkaJ5mIWNh9xC5Pg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=SUfkns)


## Code Description
This pipeline demonstrates several Apache Beam concepts:

1. Pipeline I/O (beam.io.ReadFromText): Reads data from the sample_data.txt file, creating an initial PCollection of strings.
2. Composite Transform (ProcessData): This is a custom transform that encapsulates a sequence of operations:
3. SplitData (beam.Map): Splits each input string by commas into a list of strings.
4. FilterApples (beam.Filter): Keeps only the elements where the third element (item name) is 'apple'.
5. ExtractTimestamp (beam.Map): This was originally intended to extract the timestamp but was modified in the composite transform to simply pass through the split elements. The actual timestamp extraction with a TimestampedValue is done in the subsequent ParDo.
6. ParDo (ExtractTimestampDoFn): A custom DoFn is used to process each element. It parses the timestamp string from the element and associates a TimestampedValue with each element, which is necessary for windowing. Elements that cannot be processed are skipped.
7. Windowing (beam.WindowInto(FixedWindows(30 * 60))): Groups elements into fixed windows of 30 minutes based on their timestamps. This allows for processing elements within specific time intervals.
8. Map (ExtractItemAndCount): Transforms each element in the windowed data. It creates a tuple containing the original element and a count of 1. This structure is prepared for potential future aggregations (though not fully utilized in this example).
9. Filter (FilterCounts): Keeps only the elements where the count is 1. In this simple case, it effectively passes all elements from the previous step, but demonstrates the use of a filter.
10. Partition (beam.Partition): Splits the PCollection into multiple PCollections based on a partitioning function (partition_by_item). The partition_by_item function determines which output PCollection an element belongs to based on the item name (apple, banana, or other).
11. Map (Printing Partitions): Separate Map transforms are applied to each partitioned PCollection to print the elements belonging to each partition.
This example illustrates how these different transforms can be chained together to build a data processing pipeline in Apache Beam.
