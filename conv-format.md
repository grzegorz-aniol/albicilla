```json
{
  "messages": [
    {
      "role": "user",
      "content": "Ile wyniesie indywidualna kwota napiwku dla każdej osoby w naszej grupie archeologów, jeśli całkowity rachunek za kolację wynosi 250 złotych, a w grupie jest nas pięcioro?"
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"calculate_tip_split\", \"arguments\": {\"total_bill\": 250, \"number_of_people\": 5}}</tool_call>"
    },
    {
      "role": "tool",
      "content": "{\"calculate_tip_split\": {\"individual_tip_amount\": 50}}"
    },
    {
      "role": "assistant",
      "content": "Indywidualna kwota napiwku dla każdej osoby w waszej grupie archeologów wyniesie 50 złotych."
    }
  ],
  "tools": [
    {
      "name": "search_recipes",
      "description": "Wyszukiwanie przepisów kulinarnych według określonych kryteriów",
      "parameters": {
        "type": "object",
        "properties": {
          "cuisine": {
            "type": "string",
            "description": "Kuchnia przepisu"
          },
          "ingredient": {
            "type": "string",
            "description": "Składnik do wyszukania w przepisach"
          },
          "diet": {
            "type": "string",
            "description": "Ograniczenie dietetyczne dla przepisów"
          }
        },
        "required": [ ]
      }
    },
    {
      "name": "calculate_tip_split",
      "description": "Obliczanie indywidualnej kwoty napiwku dla grupy",
      "parameters": {
        "type": "object",
        "properties": {
          "total_bill": {
            "type": "number",
            "description": "Całkowita kwota rachunku"
          },
          "number_of_people": {
            "type": "integer",
            "description": "Liczba osób w grupie"
          }
        },
        "required": [
          "total_bill",
          "number_of_people"
        ]
      }
    },
    {
      "name": "search_recipes",
      "description": "Wyszukiwanie przepisów na podstawie preferencji użytkownika",
      "parameters": {
        "type": "object",
        "properties": {
          "cuisine": {
            "type": "string",
            "description": "Preferowana kuchnia"
          },
          "ingredients": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "Składniki do uwzględnienia w przepisach"
          },
          "max_time": {
            "type": "integer",
            "description": "Maksymalny czas przygotowania w minutach"
          }
        },
        "required": [
          "cuisine",
          "ingredients"
        ]
      }
    },
    {
      "name": "recommend_books",
      "description": "Rekomendacja książek",
      "parameters": {
        "type": "object",
        "properties": {
          "genre": {
            "type": "string",
            "description": "Preferowany gatunek książek"
          },
          "max_price": {
            "type": "number",
            "description": "Maksymalna cena, jaką użytkownik jest skłonny zapłacić"
          }
        },
        "required": [
          "genre",
          "max_price"
        ]
      }
    },
    {
      "name": "calculate_loan_emi",
      "description": "Oblicz równą miesięczną ratę spłaty kredytu na podstawie kwoty kredytu, oprocentowania i okresu kredytowania",
      "parameters": {
        "type": "object",
        "properties": {
          "loan_amount": {
            "type": "number",
            "description": "Kwota kredytu"
          },
          "interest_rate": {
            "type": "number",
            "description": "Oprocentowanie roczne"
          },
          "tenure": {
            "type": "integer",
            "description": "Okres kredytowania w miesiącach"
          }
        },
        "required": [
          "loan_amount",
          "interest_rate",
          "tenure"
        ]
      }
    },
    {
      "name": "create_task",
      "description": "Utwórz nowe zadanie w liście rzeczy do zrobienia",
      "parameters": {
        "type": "object",
        "properties": {
          "title": {
            "type": "string",
            "description": "Tytuł zadania"
          },
          "due_date": {
            "type": "string",
            "format": "date",
            "description": "Termin wykonania zadania"
          }
        },
        "required": [
          "title",
          "due_date"
        ]
      }
    }
  ]
}
```