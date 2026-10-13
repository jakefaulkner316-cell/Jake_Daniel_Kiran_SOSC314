# Labelling Guide: Advertisement vs Non-Advertisement

*SOSC 314 final phase, task a. Used for the second hand-labelled sample (about 150 records, all decades).*

## 1. Why we are labelling

Our sentiment scores only describe advertising language if the records are advertisements. The Week 4 filter caught 2 of 37 non-ads in Kiran's first 200 labels (recall 5%). The case-insensitive fix raised this to 13 of 37 (35%). Those labels all come from 2012 to 2014. This sample covers every decade from 1948 to 2014, so we can:

- measure how well each filter works in every decade
- train and test a text classifier (baseline blocklist vs improved approaches)
- show that the null result is not caused by non-advertising text

## 2. The one question

> **Is this record a paid message from an outside organisation trying to persuade readers?**

If yes, it is an **ad**. If no, it is a **non-ad**. Then choose the most specific category code below.

Decide from the **text only**. The brand field can be wrong: some of The Economist's own promotions are filed under unrelated brand names.

## 3. Category codes

Choose exactly one code per record. The `is_ad` column fills itself from the code.

### Ads (`is_ad` = 1)

| Code | Name | What it looks like | Example cues |
|---|---|---|---|
| **A1** | Persuasive ad | Promotes a product, service, brand or company image. Has a selling message. | Slogans, product names, "call now", prices, "the world's leading..." |
| **A2** | Recruitment ad | An outside organisation advertises a job. | "We are seeking...", "Applications are invited for the post of...", salary, closing date |
| **A3** | Other outside promotion | Events, courses, conferences, tourism, tenders and appeals by an outside organisation. | "MBA programme", "conference in Geneva", "visit Malaysia", "invitation to tender" |

### Non-ads (`is_ad` = 0)

| Code | Name | What it looks like | Example cues |
|---|---|---|---|
| **N1** | Editorial | Articles, letters, contents pages, captions, news text. | "Letters are welcome", "Sir, ...", article prose, page headers |
| **N2** | Economist house content | The Economist promoting itself: subscriptions, offers, its own books, events, the Economist Intelligence Unit. | "Subscription service", "Offer to readers", "Economist Intelligence Unit", "The Economist Diary" |
| **N3** | Financial or legal notice | Text published because a rule or contract requires it, not to persuade. | "This announcement appears as a matter of record only", "Notice is hereby given", AGM notices, balance sheets, results tables, bond tombstones, lists of bank branches, prospectus text |
| **N4** | Mixed or multi-item | Several unrelated items run together, or a long supplement mixing ads and articles, where no single type covers more than half the text. | Two brand names with unrelated copy; a sponsored "special report" that is mostly articles |
| **N5** | Unreadable OCR | Too garbled to tell what it is. | Strings of broken characters, single letters, numbers only |

## 4. Decision rules for hard cases

1. **Majority rule.** If a record mixes types, label it by whatever makes up more than half the words. Use N4 only when no type passes half.
2. **The duty test for financial text.** Ask: *would this text exist if the company had no legal or reporting duty?*
   - A tombstone, AGM notice or results table exists because of a duty, so it is **N3**.
   - "Another record year: thank you to our customers" exists to persuade, so it is **A1**, even if it includes figures.
3. **Bank branch lists.** A list of addresses with no selling message is **N3**. A branch list under a clear slogan or service pitch is **A1**.
4. **Economist content is always N2,** even when it looks like a normal ad. This includes Economist books, events, conferences and the Intelligence Unit.
5. **Advertorials and sponsored reports.** If the text reads as an article paid for by an advertiser, use the majority rule: mostly article prose is N4, mostly product copy is A1.
6. **Noisy but readable OCR.** If you can still tell what the record is, label what it is. Use N5 only when you truly cannot tell.
7. **Long records.** The sheet shows the first 5,000 characters. If `text_truncated` = 1, judge from what you can see and add a note.
8. **Unsure.** Pick the best code, set `confidence` to 1 and write a short note. Do not skip rows.
9. **N1 means The Economist's own journalism, nothing else.** *(Added after the first agreement check.)* An ad written as a story, an interview or a descriptive paragraph is still an ad if an outside organisation wrote it to promote itself. Look for its name, a product, an address, "write to", "visit" or a slogan. Examples from the sample: HP describing its calculators (L075), Kellogg describing its Iran project (L061) and a Paris flat development (L090) are all A1, not N1.
10. **The N5 threshold.** *(Added after the first agreement check.)* Heavy noise alone does not make a record N5. If you can still name the advertiser and spot a selling line (for example "The helpful bank... Canada's International Bank" inside garbled text, L006), label what it is with confidence 1 or 2. Use N5 only when you cannot tell who or what the record is.

## 5. Columns you fill in

| Column | What to enter |
|---|---|
| `category` | One code from the dropdown: A1, A2, A3, N1, N2, N3, N4, N5 |
| `confidence` | 3 = sure, 2 = fairly sure, 1 = guessing |
| `notes` | Optional. Required when confidence is 1 or the text is truncated. |

Leave every other column unchanged. Do not sort the sheet or delete rows. `label_id` and `row_id_0based` are how we join the labels back to the corpus.

`row_id_0based` starts at 0. Kiran's first file used a `row_id` starting at 1. The new column name is there so that mistake cannot happen again.

## 6. Workflow

1. **Calibrate first.** Before starting, Kiran and Daniel each label the first 5 rows, compare answers and agree on any rule that was unclear. Add the agreed rule to section 4 of this guide.
2. **Label everything.** Kiran labels all rows in `labelling_sheet_main.xlsx`. Expect roughly 30 to 40 seconds per record, about 1.5 to 2 hours in total. Work in two sittings.
3. **Double-label a subset.** Daniel labels the 30 rows in `labelling_sheet_second_coder.xlsx` without looking at Kiran's answers. The rows appear in a different order.
4. **Measure agreement.** We compute Cohen's kappa on `is_ad` and on `category`. A kappa above about 0.6 is usually read as substantial agreement.
5. **Resolve disagreements.** Discuss each disagreement and record the agreed answer in `final_category`. Keep both original answers.
6. **Return the file** to `final phase/01 ad filter tightening/labelling sheets/` with `_labelled` added to the file name.

## 7. Why the sample is not purely random

Non-ads are a minority of records, so a purely random sample of 150 would contain only about 25 to 30 of them. To see more non-ads, each decade is split into three groups and sampled separately:

- **cue:** the text contains a phrase typical of non-ads, such as "subscription", "matter of record", "notice is hereby given" or "The Economist"
- **long:** among the longest 15% of records in that decade, which often hold supplements and notices
- **other:** everything else

Each record carries a `weight` in the key file. The weight equals how many corpus records it stands for. When we report precision and recall for the whole corpus, we use these weights. Without them, the oversampled groups would make the filters look better or worse than they really are.

Labellers do not see which group a row came from, so the group cannot influence the label.

**One caution for later:** the cue phrases overlap with the filter rules we will build in task c. Rules will look strong on the cue group almost by design. The honest test of a rule is its performance on the weighted sample, and on the **other** group alone.
