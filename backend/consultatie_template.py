"""
Consultatie Report Template
Exact format as specified by the user
"""

def generate_consultatie_template(transcript: str, patient_id: str) -> str:
    """Generate consultatie report in the exact specified format"""
    
    # Get current date
    from datetime import datetime
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    template = f"""CONSULTATIEVERSLAG CARDIOLOGIE
Patiënt ID: {patient_id}
Datum: {current_date}

1. Reden van komst
Patiënt komt (voor de eerste maal (op eigen initiatief/doorgestuurd door de huisarts) owv. …/in opvolging owv. …).

2. Voorgeschiedenis
i. Persoonlijke antecedenten:
(Persoonlijke voorgeschiedenis: operaties, opnames, ziektes etc)

ii. Familiaal
- kinderen? gezond?
- prematuur coronair lijden?
- plotse dood?

iii. Beroep:

iv. Usus:
- nicotine
- ethyl
- druggebruik

v. Thuismedicatie

3. Anamnese
Algemene beschrijving van reden van komst en van de klachten. 
Retrosternale last: (ja/neen)
Kortademigheid: (ja/neen)
Hartkloppingen: (ja/neen)
Zwelling onderste ledematen: (ja/neen)
Draaierigheid/flauwtes/bewustzijnsverlies: (ja/neen)
Bij elk van de bovenstaande: beschrijving (kwalitatief: hoe voelt het), wat lokt het uit (vb. Inspanning), etc.

4. Klinisch onderzoek
Algehele aanblik: goed
Cor: regelmatig, geen souffle
Longen: zuiver
Perifeer: geen oedemen
Jugulairen: niet gestuwd

5. Aanvullend onderzoek

i. ECG op raadpleging ({current_date}):
- ritme: (sinusaal/VKF/voorkamerflutter/atriale tachycardie).
- PR: (normaal/verlengd/verkort) (...) ms.
- QRS: (normale/linker/rechter) as, (smal/verbreed met LBTB/verbreed met RBTB/verbreed met aspecifiek IVCD).
- repolarisatie: (normaal/gestoord met).
- QTc: (normaal/verlengd) (...) ms.

ii. Fietsproef op raadpleging ({current_date}):
Patiënt fietst tot (...) W waarbij de hartslag oploopt van (...) tot (...)/min ((...)% van de voor leeftijd voorspelde waarde). De bloeddruk stijgt tot (...)/(...)mmHg. Klachten: (ja/neen).
ECG tijdens inspanning toont (wel/geen) argumenten voor ischemie en (wel/geen) aritmie.

iii. TTE op raadpleging ({current_date}):
Linker ventrikel: (...)troof met EDD (...) mm, IVS (...) mm, PW (...) mm. Globale functie: (goed/licht gedaald/matig gedaald/ernstig gedaald) met LVEF (...)% (geschat/monoplane/biplane).
Regionaal: (geen kinetiekstoornissen/zone van hypokinesie/zone van akinesie)
Rechter ventrikel: (...)troof, globale functie: (...) met TAPSE (...) mm.
Diastole: (normaal/vertraagde relaxatie/dysfunctie graad 2/ dysfunctie graad 3) met E (...) cm/s, A (...) cm/s, E DT (...) ms, E' septaal (...) cm/s, E/E' (...). L-golf: (ja/neen).
Atria: LA (normaal/licht gedilateerd/sterk gedilateerd) (...) mm.
Aortadimensies: (normaal/gedilateerd) met sinus (...) mm, sinotubulair (...) mm, ascendens (...) mm.
Mitralisklep: morfologisch (normaal/sclerotisch/verdikt/prolaps/restrictief). insufficiëntie: (...), stenose: geen.
Aortaklep: (tricuspied/bicuspied), morfologisch (normaal/sclerotisch/mild verkalkt/matig verkalkt/ernstig verkalkt). Functioneel: insufficiëntie: geen, stenose: geen.
Pulmonalisklep: insufficiëntie: spoor, stenose: geen.
Tricuspiedklep: insufficiëntie: (...), geschatte RVSP: ( mmHg/niet opmeetbaar) + CVD (...) mmHg gezien vena cava inferior: (...) mm, variabiliteit: (...).
Pericard: (...).

iv. Recente biochemie op datum (...)-(...)-(...):
- Hb (...) g/dL
- Creatinine (...) mg/dL en eGFR (...) mL/min.
- LDL (...) mg/dL
- nuchtere glycemie: (...)
- HbA1c (.../niet bepaald)
- schildklierset (normaal/niet bepaald)

6. Besluit
Uw (leeftijd)-jarige patiënt werd gezien op de raadpleging cardiologie op {current_date}. Wij weerhouden volgende problematiek:

i. (reden van komst)
Patiënt functioneert momenteel in NYHA (I/II/III/IV).
Er zijn (klachten/geen klachten) die zouden kunnen wijzen op coronair lijden

Verder is hij/zij gekend met:
- coronair:
- ritmologisch:
- structureel/hartfalen:

ii. Aandacht dient te gaan naar optimale cardiovasculaire preventie met
- Vermijden van tabak.
- Tensiecontrole( met streefdoel <130/80 mmHg/. Geen gekende hypertensie). Graag uw verdere opvolging. 
- LDL-cholesterol < (100/70/55) mg/dL. Actuele waarde (...) mg/dL (aldus goed onder controle/waarvoor opstart /waarvoor intensifiëring van de statinetherapie naar ).
- (Adequate glycemiecontrole met streefdoel HbA1c <6.5%/Geen argumenten voor diabetes mellitus type II).
- Lichaamsgewicht: BMI 20-25 kg/m² na te streven.
- Lifestyle-advies: mediterraan dieet arm aan verzadigde of dierlijke  vetten en focus op volle graan producten, groente, fruit en vis. Zoveel lichaamsbeweging als mogelijk met liefst dagelijks beweging en 3-5x/week ged. 30 min een matige fysieke inspanning.

7. Beleid
i. Medicatiewijzigingen
ii. Follow-up
"""
    
    return template

