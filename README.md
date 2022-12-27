Forests of randomized trees

Modulul sklearn.ensemble include doi algoritmi de mediere bazați pe arbori de decizie aleatoriu: algoritmul RandomForest și metoda Extra-Trees. Ambii algoritmi sunt tehnici de perturbare și combinare [B1998] special concepute pentru copaci.
Aceasta înseamnă că un set divers de clasificatori este creat prin introducerea aleatoriei în construcția clasificatorului. Predicția ansamblului este dată ca predicție medie a clasificatorilor individuali.

Ca și alți clasificatori, clasificatorii de pădure trebuie să fie echipați cu două matrice: o matrice rară sau densă X de formă (n_samples,
n_features) care deține mostrele de antrenament și o matrice Y de formă (n_samples), care conține valorile țintă (etichete de clasă) pentru mostrele de antrenament:

:[
    >>> from sklearn.ensemble import RandomForestClassifier
>>> X = [[0, 0], [1, 1]]
>>> Y = [0, 1]
>>> clf = RandomForestClassifier(n_estimators=10)
>>> clf = clf.fit(X, Y)
]

La fel ca arborii de decizie, pădurile de arbori se extind și la probleme cu mai multe rezultate (dacă Y este o matrice de formă (n_samples, n_outputs)).

Random Forests
În pădurile aleatorii (vezi clasele RandomForestClassifier și RandomForestRegressor), fiecare arbore din ansamblu este construit dintr-un eșantion extras cu înlocuire (adică un eșantion bootstrap) din setul de antrenament.

În plus, la împărțirea fiecărui nod în timpul construcției unui copac,
cea mai bună împărțire este găsită fie din toate caracteristicile de intrare, fie dintr-un subset aleatoriu de dimensiune max_features. (Consultați instrucțiunile de reglare a parametrilor pentru mai multe detalii).

Scopul acestor două surse de aleatorie este de a scădea varianța estimatorului de pădure. Într-adevăr, arborii de decizie individuali prezintă de obicei o variație mare și tind să se supraadapte.
Aleatoritatea injectată în păduri produce arbori de decizie cu erori de predicție oarecum decuplate. Luând o medie a acestor predicții, unele erori se pot anula. Pădurile aleatorii realizează o variație redusă prin combinarea diverșilor copaci, uneori cu prețul unei ușoare creșteri a părtinirii.
În practică, reducerea varianței este adesea semnificativă, rezultând astfel un model general mai bun.

Spre deosebire de publicația originală [B2001], implementarea scikit-learn combină clasificatorii făcând o medie a predicției probabilistice, în loc să lase fiecare clasificator să voteze pentru o singură clasă.

Extremely Randomized Trees

În arborii extrem de randomizați (vezi clasele ExtraTreesClassifier și ExtraTreesRegressor), aleatorietatea merge un pas mai departe în modul în care sunt calculate împărțirile. Ca și în pădurile aleatorii, se folosește un subset aleatoriu de caracteristici candidate, dar în loc să se caute pragurile cele mai discriminatorii,
pragurile sunt trase aleatoriu pentru fiecare caracteristică candidată și cel mai bun dintre aceste praguri generate aleatoriu este ales ca regulă de împărțire. Acest lucru permite de obicei reducerea variației modelului puțin mai mult, în detrimentul unei creșteri puțin mai mari a părtinirii:

:[>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.datasets import make_blobs
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> from sklearn.tree import DecisionTreeClassifier

>>> X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
...     random_state=0)

>>> clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
...     random_state=0)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> scores.mean()
0.98...

>>> clf = RandomForestClassifier(n_estimators=10, max_depth=None,
...     min_samples_split=2, random_state=0)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> scores.mean()
0.999...

>>> clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
...     min_samples_split=2, random_state=0)
>>> scores = cross_val_score(clf, X, y, cv=5)
>>> scores.mean() > 0.999
True ]

Parameters

Principalii parametri care trebuie ajustați atunci când utilizați aceste metode sunt n_estimators și max_features. Primul este numărul de copaci din pădure. Cu cât este mai mare, cu atât mai bine, dar și cu atât va dura mai mult pentru a calcula. În plus, rețineți că rezultatele nu vor mai fi semnificativ mai bune dincolo de un număr critic de copaci.
Aceasta din urmă este dimensiunea subseturilor aleatoare de caracteristici care trebuie luate în considerare la împărțirea unui nod. Cu cât este mai mică, cu atât este mai mare reducerea varianței, dar și creșterea părtinirii. Valorile implicite empirice bune sunt max_features=1.
0 sau echivalent max_features=Niciuna (luând întotdeauna în considerare toate caracteristicile în loc de un subset aleatoriu) pentru probleme de regresie și max_features="sqrt" (folosind un subset aleatoriu de dimensiune sqrt(n_features)) pentru sarcini de clasificare (unde n_features este numărul de caracteristici în date). Valoarea implicită a max_features=1.
0 este echivalent cu copacii tăiați în saci și se poate obține mai multă aleatorie prin setarea unor valori mai mici (de exemplu, 0,3 este o valoare implicită tipică în literatură). Rezultate bune sunt adesea obținute atunci când setați max_depth=None în combinație cu min_samples_split=2 (adică atunci când dezvoltați pe deplin copacii). Rețineți totuși că aceste valori nu sunt de obicei optime,
și ar putea avea ca rezultat modele care consumă multă memorie RAM. Cele mai bune valori ale parametrilor trebuie întotdeauna validate încrucișat. În plus, rețineți că, în pădurile aleatoare, eșantioanele bootstrap sunt utilizate în mod implicit (bootstrap=True), în timp ce strategia implicită pentru arborele suplimentar este de a folosi întregul set de date (bootstrap=False).
Atunci când se folosește eșantionarea bootstrap, eroarea de generalizare poate fi estimată pe eșantioanele lăsate în afara sau în afara sacului. Acest lucru poate fi activat setând oob_score=True.

Parallelization

În sfârșit, acest modul prezintă și construcția paralelă a arborilor și calculul paralel al predicțiilor prin parametrul n_jobs. Dacă n_jobs=k, calculele sunt partiționate în k joburi și rulează pe k nuclee ale mașinii. Dacă n_jobs=-1, atunci toate nucleele disponibile pe mașină sunt utilizate. Rețineți că din cauza inter-
Procesul de comunicare generală, accelerarea s-ar putea să nu fie liniară (adică, utilizarea k joburi nu va fi, din păcate, de k ori mai rapidă). Cu toate acestea, se poate obține o accelerare semnificativă atunci când se construiește un număr mare de copaci sau când construirea unui singur arbore necesită o perioadă destul de mare (de exemplu, pe seturi de date mari).

Exemple:
Trasează suprafețele de decizie ale ansamblurilor de arbori pe setul de date iris

Importanțele pixelilor cu o pădure paralelă de copaci

Faceți finalizarea cu estimatori multi-ieșiri

Referințe

[B2001]
Breiman, „Random Forests”, Machine Learning, 45(1), 5-32, 2001.

[B1998]
Breiman, „Clasificatorii arcului”, Analele statisticii 1998.

P. Geurts, D. Ernst. și L.
Wehenkel, „Extremely randomized trees”, Machine Learning, 63(1), 3-42, 2006.

Feature importance evaluation

Rangul relativ (adică adâncimea) unei caracteristici utilizate ca nod de decizie într-un arbore poate fi utilizat pentru a evalua importanța relativă a acelei caracteristici în ceea ce privește predictibilitatea variabilei țintă. Caracteristicile utilizate în partea de sus a arborelui contribuie la decizia finală de predicție a unei fracțiuni mai mari din eșantioanele de intrare.
Fracția așteptată a eșantioanelor la care contribuie poate fi astfel utilizată ca o estimare a importanței relative a caracteristicilor. În scikit-learn, fracția de eșantioane la care contribuie o caracteristică este combinată cu scăderea impurităților din divizarea lor pentru a crea o estimare normalizată a puterii predictive a acelei caracteristici.
Făcând media estimărilor capacității de predicție pe mai mulți arbori randomizați, se poate reduce varianța unei astfel de estimări și o poate folosi pentru selecția caracteristicilor. Aceasta este cunoscută ca scăderea medie a impurităților sau MDI. Consultați [L2014] pentru mai multe informații despre MDI și evaluarea importanței caracteristicilor cu Random Forests.

Avertisment Importanțele caracteristicilor bazate pe impurități calculate pe modelele bazate pe arbore suferă de două defecte care pot duce la concluzii înșelătoare. În primul rând, acestea sunt calculate pe statistici derivate din setul de date de antrenament și, prin urmare, nu ne informează neapărat asupra caracteristicilor care sunt cele mai importante pentru a face predicții bune asupra setului de date păstrat. În al doilea rând,
ele favorizează caracteristici de cardinalitate ridicată, adică caracteristici cu multe valori unice. Importanța caracteristicii de permutare este o alternativă la importanța caracteristicii bazate pe impurități care nu suferă de aceste defecte. Aceste două metode de obținere a importanței caracteristicilor sunt explorate în: Importanța permutației vs Importanța caracteristicilor forestiere aleatorii (MDI).

Următorul exemplu arată o reprezentare cu coduri de culori a importanței relative a fiecărui pixel individual pentru o sarcină de recunoaștere a feței folosind un model ExtraTreesClassifier.


În practică, acele estimări sunt stocate ca un atribut numit feature_importances_ pe modelul adaptat. Aceasta este o matrice cu formă (n_features,
) ale căror valori sunt pozitive și însumează 1,0. Cu cât valoarea este mai mare, cu atât este mai importantă contribuția caracteristicii de potrivire la funcția de predicție.

Exemple:

Importanțele pixelilor cu o pădure paralelă de copaci

Caracteristici importante cu o pădure de copaci

Referințe

[L2014]
G. Louppe, „Înțelegerea pădurilor aleatorii: de la teorie la practică”
, teză de doctorat, U. din Liege, 2014.

1.11.2.6. Încorporare total aleatorie a arborilor
RandomTreesEmbedding implementează o transformare nesupravegheată a datelor. Folosind o pădure de copaci complet aleatoriu, RandomTreesEmbedding codifică datele după indicii frunzelor în care ajunge un punct de date. Acest index este apoi codificat într-o manieră una din K,
conducând la o codificare binară de dimensiuni mari, rară. Această codificare poate fi calculată foarte eficient și poate fi apoi folosită ca bază pentru alte sarcini de învățare. Dimensiunea și dispersitatea codului pot fi influențate prin alegerea numărului de arbori și a adâncimii maxime pe copac. Pentru fiecare arbore din ansamblu, codificarea conține o intrare a câte una.
Dimensiunea codificării este de cel mult n_estimators * 2 ** max_depth, numărul maxim de frunze din pădure.

Deoarece punctele de date învecinate sunt mai probabil să se afle în aceeași frunză a unui arbore, transformarea efectuează o estimare implicită, neparametrică a densității.

Exemple:

Transformarea caracteristicii hashing folosind arbori total aleatori
Învățare variabilă pe cifrele scrise de mână: încorporarea liniară locală, Isomap... compară tehnicile de reducere a dimensionalității neliniare pe cifrele scrise de mână.

Transformările de caracteristici cu ansambluri de arbori compară transformările de caracteristici bazate pe arbori supravegheate și nesupravegheate.

Vezi și Tehnicile multiple de învățare pot fi, de asemenea, utile pentru a deriva non-
reprezentări liniare ale spațiului caracteristic, de asemenea, aceste abordări se concentrează și pe reducerea dimensionalității.
