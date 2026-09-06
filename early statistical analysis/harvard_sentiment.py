import pandas as pd
import matplotlib.pyplot as plt
#To read excel
harvard = pd.read_excel(
    "../Initial file imports/inquireraugmented.xls",
    engine="xlrd"
)

# Show the first few rows.
print(harvard.head())

# Show the column names.
print(harvard.columns)

# Show the size of the dictionary.
print(harvard.shape)

# Create separate lists of positive and negative words from the Harvard dictionary.
# For now, we are focusing on the main Positiv and Negativ categories,
# even though the dictionary includes many other categories that could be explored later.

positive_words = harvard[
    harvard["Positiv"].notna()
]["Entry"]

negative_words = harvard[
    harvard["Negativ"].notna()
]["Entry"]


# Remove missing values and make all words lowercase so they will
# match the lowercase advertisement text later.
positive_words = (
    positive_words
    .dropna()
    .astype(str)
    .str.lower()
)

negative_words = (
    negative_words
    .dropna()
    .astype(str)
    .str.lower()
)


# Show how many positive and negative words are in the dictionary.
print("\nNUMBER OF POSITIVE WORDS:")
print(len(positive_words))

print("\nNUMBER OF NEGATIVE WORDS:")
print(len(negative_words))


# Show a few examples from each.
print("\nFIRST POSITIVE WORDS:")
print(positive_words.head(20))

print("\nFIRST NEGATIVE WORDS:")
print(negative_words.head(20))
# Compare the number of positive and negative words
sentiment_counts = pd.Series({
    "Positive": len(positive_words),
    "Negative": len(negative_words)
})

print("\nSENTIMENT WORD COUNTS:")
print(sentiment_counts)

sentiment_counts.plot(kind="bar")

plt.title("Harvard Dictionary: Positive vs Negative Words")
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Words")

plt.savefig("harvard_positive_vs_negative.png")
plt.close()
