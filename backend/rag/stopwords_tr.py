"""Turkish function words dropped from BM25 queries.

A natural question is mostly function words. Because BM25 terms are OR'ed, a
chunk that merely shares "içinde" or "olarak" with the question becomes a
candidate, and fusion can lift it over a correct result. Removing these words
leaves the terms that actually say what the question is about.

Only the query is filtered. The index keeps every word, so a phrase that
happens to be all stopwords is still reachable, and the embedding search
always receives the untouched question.

Deliberately absent:

- Negations and words that carry meaning on their own: "değil", "yok",
  "olmaz", "olmayan". Dropping these would reverse what a question asks.
- Number words other than "bir": legislation is full of meaningful counts
  ("iki yıl", "on gün"), so only the article-like "bir" is dropped.
- Ordinary nouns that look frequent but are the subject of real questions,
  such as "süre" and "gün". "kaç gün içinde" is genuinely a question about
  days; only "kaç" and "içinde" are noise. BM25's IDF already discounts
  terms that appear everywhere.
"""

STOPWORDS_TR = frozenset(
    """
    acaba ama ancak artık ayrıca aynı az
    bana bazen bazı belki ben benden beni benim
    bile bir biraz birçok biri birkaç birlikte biz bize bizi bizim
    bu buna bunda bundan bunlar bunları bunların bunu bunun burada bütün
    çok çünkü
    da daha de diğer diye dolayı
    eğer en
    fakat
    gibi göre
    hangi hangisi hatta hem henüz hep hepsi her herhangi herkes hiçbir
    için içinde ile ilgili ise itibaren
    kadar karşı karşın kendi kez ki kim kime kimi kimin kimse
    mi mı mu mü
    nasıl ne neden nedenle nerede nereden nereye neyi niçin niye
    o olan olarak oldu olduğu olduğunu olması olsa olsun olup olur olursa
    ona onda ondan onlar onları onların onu onun
    öyle önce
    rağmen
    sadece sen senin siz size sizin sonra
    şey şeyler şu şunlar şunu
    tarafından tüm
    üzere üzerine
    var vardır ve veya
    ya yani yerine yine yoksa
    zaten
    """.split()
)


def is_stopword(word: str) -> bool:
    """Whether a query word is a function word, ignoring case.

    str.lower() is not enough on its own: "İÇİNDE".lower() keeps a combining
    dot above and would not match "içinde", so İ is mapped to i first.
    """
    return word.replace("İ", "i").lower() in STOPWORDS_TR
