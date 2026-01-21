# Wsparcie w pozyskiwaniu danych z sesji agentowych do uczenia Bielika

## O co chodzi?

Bielik, aby lepiej współpracował w systemach agentowych, powinien być trenowany na danych pochodzących z takich sesji. Potrzebujemy pozyskać takie dane.

## Co powinienem umieć, aby pomóc?

Powinieneś mieć doświadczenie z pracą z systemami agentowymi, umieć je konfigurować, mieć doświadczenie programistyczne (np. Python) i znać sposób użycia serwerów MCP.

## Jakie to powinny być dane?

Dane powinny być wysokiej jakości, realistyczne, ale przede wszystkim bardzo urozmaicone. Prompty nie powinny zawierać błędów, literówek ani treści słabej jakości.

Przykład tematu dla wzorcowej sesji:
* Planuję podróż do Gruzji, korzystam z narzędzi do wyszukiwania lotów, hoteli, atrakcji turystycznych i Google Maps, instruuję agenta, aby zaplanował mi 7-dniową wycieczkę, podaję instrukcje, zgłaszam komentarz itd., itp.
* Zbieramy wymagania funkcjonalne od użytkownika, rozpoznajemy jego oczekiwania, weryfikujemy je z politykami firmy oraz rozwoju produktu, sprawdzamy czy były już podobne zgłoszenia lub wymagania, odpowiadamy użytkownikowi.

Przykład słabej sesji:
* Podaj aktualną temperaturę na wyspach Hula-Gula.

## Co masz na myśli przez „sesję”?

Zapis rozmowy między użytkownikiem a agentem, z informacjami o wywołanych narzędziach. Sesja może zawierać jednorazową interakcję z agentem (użytkownik pyta, agent sprawdza i odpowiada) albo być wielokrotną wymianą zdań i poleceń, przeplataną wywołaniami narzędzi.

## Ile danych jest potrzebne?

Najważniejsze to jakość i rozmaitość scenariuszy. Ilość jest aktualnie sprawą drugorzędną.

## W jakim języku powinny być dane?

Na pewno ważny jest język polski i angielski. Jeżeli masz możliwość pozyskania danych w innych językach europejskich, wspieranych przez Bielika, będzie to dodatkowy atut.

## Jak można pozyskać dane?

Jeśli twój agent korzysta z modeli poprzez OpenAI-compatible Chat Completions API, możesz użyć serwera proxy, który wpinasz pomiędzy twojego agenta a prawdziwy model. Proxy znajdziesz tutaj: https://github.com/grzegorz-aniol/albicilla. Tam znajdziesz instrukcje, jak włączyć proxy i skonfigurować endpoint w swoim agencie.

Proxy loguje wszystkie requesty i odpowiedzi. Do trenowania potrzebujemy jednak dane z całej sesji w ujednoliconym formacie i z zapisem tool calling w taki sposób, w jaki oczekuje tego Bielik podczas trenowania.

Dlatego „albicilla” ma również CLI do konwersji logów do finalnego formatu — i o takie dane nam chodzi!

## Czy można pozyskać dane w inny sposób?

Tak. Natomiast dane powinny być w formacie opisanym w repo Albicilla: https://github.com/grzegorz-aniol/albicilla/blob/main/conv-format.md.

Jeżeli masz możliwość wyeksportowania takich danych na potrzeby Bielika z prawdziwych systemów, zgłoś się do fundacji — chętnie pozyskamy takie dane w sposób formalny.

## Na co powinienem zwrócić uwagę?

Dane z sesji nie powinny zawierać danych wrażliwych, prywatnych, poufnych.

Udostępniając dane, zgadzasz się na ich użycie w procesie trenowania modeli fundacji SpeakLeash.

Ważne: proxy nie loguje kluczy API — jedynie treść requestów i odpowiedzi jest zapisywana do plików.

## Jak przekazać dane?

Dane powinny być w formacie JSONL.

Wrzuć je np. na Dysk Google lub Dropbox i udostępnij w trybie do odczytu. Skontaktuj się na Discordzie SpeakLeash ze mną (GregA).

## Mam więcej pytań?
Kontakt ze mną (jak wyżej).
