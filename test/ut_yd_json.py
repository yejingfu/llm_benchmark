#from openai import OpenAI
import argparse
import time
import aiohttp
import asyncio
import json
import random
from collections import OrderedDict

SYSTEM_PROMPT= """
Generate responses exclusively in valid JSON format with no additional text. Each response must be a fully structured JSON object without explanations, introductions, or comments. Ensure that the JSON is complete and valid, ready to be used directly in a JSON validator.
"""
PROMPT_PREFIX = """
Generate a minified JSON object with the structure:
{"formatted_description": "DESCRIPTION"}
Replace DESCRIPTION with an HTML-formatted version of the following text that:
Uses the original language ({lang_code})
Starts with a 30-word summary of the job description (no introductory phrases)
Restructures and reformulates the content, adding value without inventing information
Uses only basic HTML tags (no <!DOCTYPE>, <html>, <body>, <head>, or <a>)
Includes bold text and bullet points for better readability
Removes redundancies, repetitions, and blocked content (emails, URLs, phone numbers, social media, addresses, ID numbers, dates, hashtags)
Excludes words: urgent, earn, apply now (and their synonyms)
Avoids clickbait language and promotional content
Do not repeat paragraphs.
Properly escapes all strings for JSON, using "\n" for line breaks
Contains no unescaped control characters or newlines within JSON strings
Do not wrap the JSON object in an array; return only the JSON object as specified.
The JSON output is minified: do not include any unnecessary whitespace, line breaks, or indentation outside of string values.
Remove or replace any Unicode characters that may cause HTML errors.
Do not include explanations or additional text or markup outside the JSON object.
Original text to rewrite:

"""
PROMPT_SUFFIX = """

Ensure the output is a single, minified JSON object with properly escaped content and no problematic Unicode characters.
"""

USER_PROMPTS = [
{"lang": "fr", "content":
"""
<p>Nous recherchons actuellement un ExpertComptable Calais.</p><p>Salaire : Entre 50 000 et 60 000 brut annuel assorti davantages tels que des tickets restaurant une mutuelle des jours de RTT une flexibilit horaire la participation au Comit Social et conomique (CSE) un statut cadre des chques cadeaux pour Nol et les vacances ainsi que la possibilit de tltravail occasionnel selon les besoins.</p><p>Contrat : CDI cadre sur une base de 39 heures par semaine.</p><p>DESCRIPTION DU POSTE</p><p>Ce poste a pour vocation de progresser vers un statut dassoci (association dans le cadre dun LBO).</p><p>Vous serez responsable de la gestion autonome dun portefeuille clients constitu de TPE/PME en vous appuyant sur vos comptences techniques en fiscalit et en comptabilit.</p><p>Vos missions comprendront ltablissement de prvisionnels de bilans de liasses fiscales de comptes de rsultat ainsi que la gestion des dclarations dimpts sur le revenu.</p><p>Selon votre degr dautonomie vous pourrez tre soutenu(e) dans un premier temps par lun des expertscomptables associs. Votre mission principale consistera encadrer une quipe de 2 3 collaborateurs et fournir des conseils personnaliss un portefeuille de clients plus importants.</p><p>Vous participerez galement au dveloppement du cabinet aux cts des expertscomptables associs (actuellement au nombre de 4).</p><p>PROFIL RECHERCH</p><p>Idalement titulaire du Diplme dExpertise Comptable (DEC) ou en voie de lobtenir ou encore mmorialiste.</p><p>Vous justifiez dau moins 3 ans dexprience dans un cabinet dexpertise comptable.</p><p>Vous matrisez les outils informatiques et possdez de solides comptences en logiciels de comptabilit et de gestion (CRM).</p>
"""
}, {"lang": "en", "content":
"""
<h3>Job Description</h3><p>Go Healthcare is seeking a travel nurse RN ICU - Intensive Care Unit for a travel nursing job in Chino, California.</p>Job Description & Requirements<ul><li>Specialty: <b>ICU - Intensive Care Unit</b></li><li>Discipline: <b>RN</b></li><li>Start Date: <b>11/11/2024</b></li><li>Duration: <b>52 weeks</b></li><li><b>36 hours per week</b></li><li>Shift: <b>12 hours, nights</b></li><li>Employment Type: <b>Travel</b></li></ul><p>Go Healthcare Job ID #874425. Pay package is based on 12 hour shifts and 36 hours per week (subject to confirmation) with tax-free stipend amount to be determined.</p>About Go Healthcare<p>From the first conversation with your Go Recruiter to your last day of your assignment, you can tell we are a different travel nurse company. Simply put, we do everything to make you happy. We center our company on YOU, what you want and what you need. Discover the Go Healthcare Staffing difference \u2013 we are a Joint Commission Certified and Women Owned and Operated Travel Staffing Company who puts nurses first </p>Benefits<ul><li>Weekly pay</li><li>Holiday Pay</li><li>Guaranteed Hours</li><li>401k retirement plan</li><li>Mileage reimbursement</li><li>Referral bonus</li><li>Medical benefits</li><li>Dental benefits</li><li>License and certification reimbursement</li><li>Vision benefits</li><li>Life insurance</li></ul>
"""
}, {"lang": "es", "content":
"""
<p>**RENTOKIL MÉXICO**</p><p>**Servicio de Control de Plagas en Planta / Tultepec.**</p><p>**Somos una Empresa Inclusiva**</p><p>**No discriminamos en razón de género, identificación, raza, religión o alguna otra.**<br>- Balance Urbano en Control de Plagas, no te solicitará ningún pago o retribución alguna para participar en nuestro proceso de reclutamiento y selección. Tampoco te solicitará documento originales para tu contratación._<br>- Somos la empresa líder a nível mundial en control de plagas con presencia en 90 países, 57,700 colegas y 4.9 millones de clientes._<br>- **Únete a nuestro gran equipo Rentokil y haz carrera con nosotros como Técnico base especializado en control de plagas, en RENTOKIL somos protectores de personas, mejoradores del mundo y preservadores de nuestro planeta.**_<br>- Si quieres ser parte de la empresa más grande del mundo en su giro,_ comunícate con nosotros_</p><p>**Ofrecemos**:</p><p>Contratación inmediata y directa con la empresa.<br>- Sueldo base $8,700<br>- $500 en vales de despensa (fijos).Los vales de despensa se depositan en un monedero electrónico y no causan impuestos.<br>- Capacitación pagada<br>- Seguro de vida<br>- Prestaciones de ley desde el primer día de tu contratación12 días de vacaciones<br>- Prima vacacional del 25%<br>- 15 días de aguinaldo<br>- Plan de carrera<br>- Programa de premios<br>- Programa opcional para referir clientes o colaboradores<br>- Uniformes, botas y equipo de protección sin costo.<br>- Celular Utilitario<br>- Oportunidades de generar horas extras, prima dominical dependiendo de las necesidades de la empresa, por otro lado si sabes manejar y cuentas con licencia de carga, puedes generar ingresos adicionales por trabajo extra en otras plantas si se requiere.</p><p>NOTA: El teléfono y los insumos para el trabajo son sólo para uso con fines laborales, recuerda que cualquier daño o pérdida parcial o total por uso fuera de políticas es responsabilidad del colaborador</p><p>No te será requerido pago o retribución alguna para participar en ninguna de nuestras etapas del proceso de selección, además, en ningún momento te requerimos algún documento o carta original para tu contratación.<br>- **Descripción de nuestra vacante**_</p>
<p>La vacante Técnico de Servicio en Base es para trabajar en actividades de Control de Plagas en una planta de nuestro cliente de acuerdo a una programación de días y horarios preestablecidos. El Técnico brinda servicio y asesoría a nuestros clientes. Si te gusta ayudar, servir, asesorar y tratar con muchas personas, con nosotros puedes lograr más.</p><p>Los servicios básicos que incluyen el control de plagas, son revisión y cambio de trampas para roedores, trampas para insectos voladores, involucra agacharse, subir escaleras y hacer largos recorridos dentro de las instalaciones de nuestros clientes, si te gusta mantenerte en forma este es un excelente lugar para ti.</p><p>**El horario de trabajo**: Será de 8 horas diaria/Sábado medio día</p><p>**El perfil que buscamos**:<br>**Escolaridad**: Preparatoria concluida (excelente ortografía y saber calcular porcentajes o regla de 3)</p><p>**Edad**:A partir de 24 años de edad hasta 55 (excelente estado de salud)</p><p>**Vocación**:Que tengas un interés auténtico por el trabajo, no tener fobias.</p><p>**Aviso de Privacidad**</p><p>Rentokil se reserva el derecho de cambiar este Aviso de Privacidad en cualquier momento, poniendo a tu disposición tales modificaciones a través de cualquiera de los siguientes</p><p>medios: (i) anuncios visibles en nuestros establecimientos, (ii) oficina de Administración de Personal</p><p>**Para las información por medio de WhatsApp o llamada al 5524034691 LIC. ITZEL GARCIA.**</p><p>Tipo de puesto: Tiempo completo</p><p>Salario: $8,700.00 al mes</p><p>Horario:</p><p>Turno de 8 horas</p><p>Horario laboral:</p><p>Tiempo completo</p><p>Prestaciones:</p><p>Horarios flexibles<br>- Seguro de vida<br>- Teléfono de la empresa<br>- Uniformes gratuitos<br>- Vales de despensa</p><p>Lugar de trabajo: Empleo presencial</p>
"""
}, {"lang": "nl", "content":
"""
Zorgwerk bemiddelt uitzendmedewerkers en zzp'ers in de <b>zorg</b>, onderwijs, het sociaal domein en <b>kinderopvang</b> sinds <b>1997</b>.Wij zijn er voor <b>professionals</b> op alle niveaus, variërend van <b>gastvrouwen/-heren</b>, schoonmaakmedewerkers, helpenden, <b>verzorgenden</b> en <b>verpleegkundigen</b> tot <b>pedagogisch medewerkers</b>, begeleiders en <b>docenten</b>.Wekelijks zijn ruim <b>5000 professionals</b> via Zorgwerk aan het werk. Met de super gebruiksvriendelijke Zorgwerk app regel je eenvoudig de leukste diensten en opdrachten bij jou in de <b>regio</b>. Bij ons sta jij als <b>medewerker</b> of zzp'er centraal en zijn we er voor jou, <b>7 dagen per week</b>, om je te ondersteunen in je <b>werk</b> en <b>carrière</b>.<h3>Zorgwerk is het platform dat je helpt bij het vinden van de fijnste diensten en opdrachten. Excellente service. Wekelijks betaald.</h3><p>Werk via Zorgwerk als <b>uitzendmedewerker</b>, <b>zzp'er</b> of <b>interim manager</b> in de <b>zorg</b>, <b>kinderopvang</b>, <b>onderwijs</b> of het <b>sociaal domein</b>. Bepaal zélf met de geweldige Zorgwerk app waar, wanneer en hoe vaak je werkt.</p><p>Reageer op een vacature of meld jezelf direct aan en krijg toegang tot al het beschikbare werk in de overzichtelijke Zorgwerk app.</p><h3> Word jij de sprankelende ster in de Kinderopvang? </h3><br><br>
<b>Ben je op zoek naar flexibel werk als BSO/KDV Pedagogisch Medewerker in Zeeland en omgeving Brabant-Noord?</b> Bij Zorgwerk bieden we jou de mogelijkheid om te werken bij diverse opdrachtgevers op het gebied van Kinderopvang en voor verschillende doelgroepen, waaronder BSO.<br><br>Met onze handige Zorgwerk app heb je altijd en overal toegang tot leuke diensten en opdrachten bij <b>honderden opdrachtgevers</b> in diverse sectoren. Of je nu op zoek bent naar tijdelijk werk, extra diensten of een flexibele bijbaan, bij Zorgwerk vind je de mogelijkheden die bij jou passen.<br><br><b>Heb je de benodigde kwalificaties en ervaring als BSO/KDV Pedagogisch Medewerker</b> en ben je enthousiast geworden om aan de slag te gaan in Zeeland en omgeving? Meld jezelf online aan en word onderdeel van ons team van zorgprofessionals die flexibel werken bij diverse opdrachtgevers. Wij heten je van harte welkom bij Zorgwerk.<ul><li>Je hebt een diploma dat je kwalificeert voor de functie pedagogisch medewerker voor dagopvang of BSO/NSO volgens de CAO Kinderopvang, zoals SPW 3/4 (MBO), HBO Pedagogiek of PABO.</li> <li>Je hebt ervaring in de dagopvang of op de buitenschoolse opvang.</li> <li>Je bezit goede communicatieve vaardigheden en kan zowel zelfstandig als in een team functioneren.</li> <li>Je staat ingeschreven in het Personenregister Kinderopvang of je bent bereid je in te schrijven. Hiervoor dien je een VOG aan te vragen (de VOG vergoeden wij na je eerst gewerkte dienst).</li> <li>VVE-certificaat en kwalificatie taaleis 3F voor voorschoolse educatie is een pré.</li> <li>Je bent enthousiast om via Zorgwerk aan het werk te gaan.</li> <li>Je beschikt over een sterk ontwikkeld verantwoordelijkheidsgevoel.</li> <li>Zeer betrouwbaar, de opdrachtgever kan op jou rekenen.</li> <li>Je komt representatief over voor de kinderopvang, qua kledingkeuze en gedrag.</li> <li>Je hebt affiniteit met Kinderopvang en BSO.</li> <li>Je kan werken als zzp'er of in loondienst bij Zorgwerk, wat jij wilt.</li> <li>Je kan werken als zzp'er of in loondienst bij Zorgwerk. Gegarandeerd wekelijks betaald</li> <li>Goed salaris in loondienst, als zzp'er bepaal je jouw eigen uurtarief.</li> <li>Afwisseling: werk wekelijks bij uiteenlopende opdrachtgevers in diverse sectoren.</li> <li>Alle vrijheid: jij bepaalt zélf met de app waar, wanneer en hoe vaak je werkt.</li> <li>Geweldige community: vind vrienden met de app en je staat er nooit alleen voor.</li></ul>
"""
}, {"lang": "es", "content":
"""
<p>** EN GLEZMO TRANSPORTES BUSCAMOS ALMACENISTA **<br>- **REQUISITOS**:_<br>- Preparatoria concluida<br>- Conocimientos en almacen<br>- Licencia de chofer o motociclista.<br>- Conocimientos de office<br>- Experiência mínima 1 año en puesto similar.<br>- **FUNCIONES**:_<br>- Control entradas y salidas de mercancia en sistema<br>- Control de inventarios<br>- Compras de insumos<br>- Apoyo en actividades administrativas.<br>- Limpieza y orden de almacen.<br>- Notificar al jefe de mantenimiento sobre irregularidades de material detectado en la recepción o acomodo y en la documentación correspondiente.<br>- Apoyar en las demás actividades que se le encomienden y que se deriven de la propia naturaleza del puesto y del almacén.<br>- **HABILIDADES**_<br>- Buena presentación<br>- Puntualidad<br>- Disponibilidad de horario<br>- Orientado a resultados<br>- Trabajo en equipo<br>- Proactivo<br>- Responsable<br>- Organizado</p><p>**HORARIO**:<br>L- V 10:00 A 19:00 hrs</p><p>Sábados 9:00 a 15:00 pm</p><p>**PRESTACIONES SUPERIORES A LAS DE LEY.**</p><p>Tipo de puesto: Tiempo completo, Por tiempo indeterminado</p><p>Salario: $9,000.00 - $9,200.00 al mes</p><p>Entorno físico:</p><p>Almacén</p><p>Horario:</p><p>Turno de 8 horas</p><p>Prestaciones:</p><p>Caja de ahorro<br>- Opción a contrato indefinido<br>- Seguro de vida<br>- Uniformes gratuitos</p><p>Lugar de trabajo: Empleo presencial</p>
"""
}, {"lang": "es", "content":
"""
<p>_**PETSTAR SOLICITA**:_</p><p>**AUXILIAR DE COMUNICACIÓN**</p><p>**Perfil**:</p><p>Experiência de 1 año generando contenidos de valor para otras empresas y/o organizaciones.<br>- Disponibilidad para viajar.<br>- Disponibilidad para acudir a oficina central (CDMX) y planta de reclicado (TOLUCA)</p><p>**EXPERIENCIA**:</p><p>Manejo asertivo de planes de trabajo para la creación de contenido diferenciado para las distintas redes sociales.<br>- Indicadores de redes sociales para FB, X, Instagram, Tik Tok, LinkedIn.</p><p>**OFRECEMOS**:</p><p>Sueldo competitivo<br>- Vales de despensa<br>- Prestaciones superiores a la ley</p><p>**ACTIVIDADES PRINCIPALES**:<br>*</p><p>1. Apoyar en la creación de una estrategia de contenidos (videos, imágenes, reels, notas informativas, artículos, podcast, video-blog) en función de los canales, buyer personal o keywords que ayuden a gestionar una comunidad sustentable y darle seguimiento.</p><p>2. Contribuir a la creación de historias originales que emocionen y se diferencien frente a la competencia (Storytelling).</p><p>3. Redactar contenido: teniendo en cuenta los criterios SEO y los buyer persona de cada cliente. Siempre tendrá que ser original y aportar valor.</p><p>5. Dar servicio a cada red social, generando interacción con los seguidores y dándoles respuesta a sus inquietudes.</p><p>6. Gestionar anuncios para redes sociales como Instagram o Facebook.</p><p>7. Participar en organización de eventos y/o dinámicas para crear una comunidad sustentable que nos ayude con el posicionamiento.</p><p>8. Monitorear temas de riesgo a la reputación de la empresa.</p><p>Tipo de puesto: Tiempo completo</p><p>Salario: $11,500.00 al mes</p><p>Horario:</p><p>Lunes a viernes<br>- Turno de 8 horas</p><p>Prestaciones:</p><p>Estacionamiento de la empresa<br>- Opción a contrato indefinido<br>- Seguro de vida<br>- Trabajo desde casa<br>- Vales de despensa</p><p>Lugar de trabajo: remoto híbrido en 50295, Toluca de Lerdo, Méx.</p>
"""
}, {"lang": "en", "content":
"""
<h3>Job Description</h3><p>MedPro Healthcare Allied Staffing is seeking a travel CVOR Technologist for a travel job in Springfield, Ohio.</p>Job Description & Requirements<ul><li>Specialty: <b>CVOR Technologist</b></li><li>Discipline: <b>Allied Health Professional</b></li><li>Start Date: <b>11/11/2024</b></li><li>Duration: <b>13 weeks</b></li><li><b>40 hours per week</b></li><li>Shift: <b>8 hours, days</b></li><li>Employment Type: <b>Travel</b></li></ul><p>MedPro Healthcare Staffing, a Joint Commission-certified staffing agency, is seeking a quality CVOR Technician- Cardiovascular Operating Room Technician for a contract with one of our top healthcare clients.</p> Requirements<ul><li>Graduate of an accredited School of Surgical Technology or 3 years recent experience working as a surgical technician in an acute care setting.</li><li>Current CPR provider card.</li><li>Must have a minimum of eighteen months of CVOR recent experience or completed clinical orientation specific to the surgical care of the open-heart patient population.</li><li>Knowledge of CVOR instrumentation as it relates to specific cardiac surgical procedures and other procedures as needed.</li><li>CABG and cardiac procedures.</li></ul><p>Benefits</p><ul><li>Weekly pay and direct deposit</li><li>Full coverage of all credentialing fees</li><li>Private housing or housing allowance</li><li>Group Health insurance for you and your family</li><li>Company-paid life and disability insurance</li><li>Travel reimbursement</li><li>401(k) matching</li><li>Unlimited Referral Bonuses up to $1,000</li></ul>Duties Responsibilities<ul><li>Performs treatments per protocols.</li><li>Functions proficiently in scrubbing, 2nd circulating, and 3rd person/Heart Holder roles of the CVOR.</li><li>Picks supplies for surgeries, sets up instruments and scrubs for surgical procedure, and performs instrument and sponge count accurately.</li><li>Obtain specimens, hand over to circulating nurse and verifies location.</li><li>Facilitates room turnover time.</li><li>Participates in department on-call schedule.</li></ul>
<p>About Agency</p><p>MedPro Healthcare Staffing is a Joint Commission certified provider of contract staffing services. Since 1983, we have placed nursing and allied travelers in top healthcare facilities nationwide. Join us today for your very own MedPro Experience.</p><p>If qualified and interested, please call for immediate consideration.</p><p>MedPro Staffing is an Equal Opportunity Employer. All applicants will be considered for employment without attention to race, color, religion, national origin, age, sex, disability, marital status or veteran status.</p> <p>Key Words: OR Technician, Surgical, Technologist, Cardiovascular Operating Room, CVOR</p><p>*Weekly payment estimates are intended for informational purposes only and include a gross estimate of hourly wages and reimbursements for meal, incidental, and housing expenses. Your recruiter will confirm your eligibility and provide additional details.</p><p>MedPro Job ID #a0Fcx000000X6mbEAC. Pay package is based on 8 hour shifts and 40 hours per week (subject to confirmation) with tax-free stipend amount to be determined. Posted job title: Cvor Technologist Surgical: Cvor Tech.</p>About MedPro Healthcare Allied Staffing<p>No One Cares More for Caregivers Than MedPro. Focus on your patients, we'll take care of the rest. MedPro Healthcare Staffing is a Joint Commission certified provider of temporary and contract staffing services. Since 1983, we have placed happy nursing and allied travelers in top healthcare facilities nationwide. You deserve a travel experience that's rewarding and memorable. One that allows you to DREAM big. EXPLORE often. And ACHIEVE greatness. The MedPro Experience delivers it<br></p><ul><li>Access to nationwide travel assignments</li><li>Weekly pay and direct deposit</li><li>Full coverage of all credentialing fees</li><li>Private housing or housing allowance</li><li>Group Health insurance for you and your family</li><li>Tax Free Per Diems, Housing Stipends and Travel Reimbursements</li><li>Company-paid life and disability insurance</li><li>Travel reimbursement</li><li>Access to our Clinical Nurse Liaison Team</li><li>401(k) matching</li><li>Unlimited Referral Bonuses starting at $500</li><li>Personalized gifts delivered to your door step</li></ul>Benefits<ul><li>Weekly pay</li><li>Employee assistance programs</li><li>Referral bonus</li></ul>
"""
}, {"lang": "es", "content":
"""
<p>Tienes Experiência como mesera Ven y forma parte de nuestro grupo.</p><p>En Mariscos Roque te estamos buscando</p><p>**Requisitos**:</p><p>Edad: de 19 a 35 años<br>- Te gustas los alimentos del mar<br>- Tienes buena presentación y actitud de servicio.<br>- Disponibilidad para toda la semana con un día de descanso entre semana<br>- Experiência como mesera<br>- Papeles en regla y deseos de Trabajar</p><p>Tipo de puesto: Tiempo completo</p><p>Sueldo: $250.00 al día</p><p>Horario:</p><p>Incluye fines de semana</p><p>Prestaciones:</p><p>Horarios flexibles</p><p>Tipos de compensaciones:</p><p>Propinas</p><p>Escolaridad:</p><p>Secundaria terminada (Deseable)</p><p>Experiência:</p><p>puesto similar: 1 año (Deseable)</p><p>Lugar de trabajo: Empleo presencial</p>
"""
}, {"lang": "es", "content":
"""
<p>**Vacante para la empresa Mi empleo ideal en Venustiano Carranza, Ciudad de México**:<br>FUNCIONES:</p><p>Atender y gestionar llamadas telefónicas referentes a dudas o inconvenientes que tengan los clientes</p><p>**Requisitos**:</p><p>18 a 50 años<br>- Experiência mínima de 6 meses en Call Center.<br>- Bachillerato concluido o Carreta técnica concluida (comprobable).<br>- Facilidad de palabra.<br>- Manejo de computadora intermedio.<br>- Manejo de paquetería Office básico.</p><p>Horario:</p><p>Matutino (09:00 A 03:00)<br>- Vespertino (03:00 A 09:00)</p><p>Descanso Rolado</p><p>OFRECEMOS:</p><p>Sueldo base de $9,000.00<br>- Prestaciones de Ley Inmediatas (Vacaciones, Prima vacacional, Aguinaldo, Seguro Social)<br>- Seguro de Vida<br>- Capacitación pagada.<br>- Medio tiempo.<br>- Beneficios extra por ser parte de la Empresa.</p><p>**Nível de educación deseada**:<br>Media Superior</p><p>**Nível de experiência deseada**:<br>Practicantes</p><p>**Función departamental**:<br>Atención al cliente</p><p>**Industria**:<br>Call Centers / Telemarketing</p>
"""
}, {"lang": "es", "content":
"""
<p>**Nombre Posición**: Analista de Gestoría.</p><p>**Nível de contribución**: Individual</p><p>**Ubicación**:Chihuahua</p><p>**Misión del puesto**:<br>Gestionar los permisos necesarios para la apertura y renovación de nuevos proyectos cumpliendo con los requerimientos legales solicitados por las autoridades competentes, así como garantizar la permanencia de la operación de nuestras Tiendas asegurando la actualización anual de permisos de operación.</p><p>**Principales responsabilidades**:<br>1. Establecer y mantener la relación con las autoridades competentes a nível municipal y estatal (en el territorio asignado) mediante una frecuente y eficaz comunicación para mantener y crecer la operación de la cadena.</p><p>2. Gestionar y obtener en tiempo y forma los permisos necesarios cumpliendo las regulaciones vigentes ante las distintas autoridades que permitan habilitar las aperturas, renovaciones y operación de Tiendas.</p><p>3. Habilitar a la Región y Plazas con los permisos vigentes que aseguren la exhibición y control de estos en las Tiendas.</p><p>4. Dar solución en tiempo y forma a las notificaciones y requerimientos presentadas por la autoridad en materia administrativa y regulatoria.</p><p>**Decisiones relevantes**:</p><p>Escalar oportunamente cambios en regulaciones vigentes que impactarán la operación.<br>- Buscar al interior de la empresa la plataforma correspondiente para documentar ante la autoridad las posibles soluciones técnicas de los proyectos, notificaciones o requerimientos.</p><p>**Requisitos**:</p><p>Profesionista titulado: Licenciatura terminada en Derecho, Arquitectura, Mercadotecnia o afín.<br>- Contar con licencia vigente y saber manejar estándar.<br>- Experiência en: Comunicación efectiva, Negociación y Conocimientos de la función pública y/o asociaciones<br>- Competencias**:Orientación al cliente, Relaciones efectivas, Organización y Ejecución.</p><p>Si esta oportunidad es de tu interés y cumples con los requisitos del puesto y lineamientos de movilidad, postúlate por este medio</p><p>Tipo de puesto: Por tiempo indeterminado</p><p>Beneficios:</p><p>Aumentos salariales<br>- Automóvil de la empresa<br>- Caja de ahorro<br>- Descuento de empleados<br>- Estacionamiento de la empresa<br>- Seguro de gastos médicos<br>- Seguro de gastos médicos mayores<br>- Seguro de vida<br>- Teléfono de la empresa<br>- Vacaciones adicionales o permisos con goce de sueldo<br>- Vales de despensa</p><p>Pago complementario:</p><p>Bono anual<br>- Bono de productividad</p><p>Tipo de jornada:</p><p>Lunes a viernes<br>- Turno de 8 horas<br>- Turno matutino</p><p>Lugar de trabajo: On the road</p>
"""
}, {"lang": "es", "content":
"""
Es para ayudarme a terminar de montar una piscina de poliéster enterrada. Montaje de depuradora y todo lo que conlleva.<ul><li>Indica el trabajo de albañilería que se necesita </li> Terminar de hacer una piscina de poliéster enterrada<br><br><li>¿Qué tipo de trabajos de albañilería hay que realizar? </li> Solera de hormigón o similar, pequeñas construcciones (casetas, montar barbacoas, etc.)<br><br><li>Partes del inmueble sobre las que trabajará el albañil </li> Terraza<br><br><li>Tipo de profesional que se quiere contratar </li> Albañil oficial de primera (€€€)<br><br><li>Tipo de inmueble o propiedad </li> Vivienda unifamiliar<br><br><li>Perfil del inmueble donde se realizará el trabajo de albañilería </li> En propiedad<br><br><li>¿Cuándo quieres realizar el trabajo? </li> En los próximos días<br><br><li>Horario de preferencia </li> Todo el día </ul><b>Preferencia para el servicio: </b> El mejor precio
"""
}]

DEF_TEMPERATURE = 0
DEF_TOP_P = 1
DEF_PRESENCE_PENALTY = 0
DEF_FREQ_PENALTY = 0

def get_chat_payload(req, args):
    content = PROMPT_PREFIX.replace("{lang_code}", req["lang"]) + req["content"] + PROMPT_SUFFIX
    enable_json_mode = not args.no_json
    extra_body = None
    response_format = None
    obj = {
        "temperature": DEF_TEMPERATURE,
        "top_p": DEF_TOP_P,
        "presence_penalty": DEF_PRESENCE_PENALTY,
        "frequency_penalty": DEF_FREQ_PENALTY,
        #"repetition_penalty": 1,
        "stop": ["<|eot_id|>", "<start_header_id|>", "<|end_header_id|>"],
        "model": args.model,
        "messages": [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": content,
        }],
        "max_tokens": args.max_tokens,
        "response_format": response_format
    }
    if not args.no_json:
        obj["guided_json"] = {
            "type": "object",
            "properties": {
                "formatted_description": {"title": "description", "type": "string"}
            },
            "required": ["formatted_description"]
        }
    if args.stream:
        obj["stream"] = args.stream,
    return obj

async def send_one_request_openai(req, args):
    start_time = time.time()
    content = PROMPT_PREFIX.replace("{lang_code}", req["lang"]) + req["content"] + PROMPT_SUFFIX
    enable_json_mode = not args.no_json
    extra_body = None
    response_format = None
    if not args.no_json:
        extra_body={
            #"guided_decoding_backend": "lm-format-enforcer",
            #"guided_whitespace_pattern": r"[\n\t ]*",
            #"response_format": {"type": "json_object"},
            #"response_format": {"type": "json_schema"},
            #"guided_json": {"type": "object"},
            "guided_json": {
                "type": "object",
                "properties": {
                    "formatted_description": {"title": "description", "type": "string"}
                },
                "required": ["formatted_description"]
            },
        }

    chat_completion = client.chat.completions.create(
        temperature=DEF_TEMPERATURE,
        top_p=DEF_TOP_P,
        presence_penalty=DEF_PRESENCE_PENALTY,
        frequency_penalty=DEF_FREQ_PENALTY,
        #repetition_penalty=1,
        stop=["<|eot_id|>", "<start_header_id|>", "<|end_header_id|>"],
        stream=args.stream,
        model=args.model,
        messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": content,
        }],
        max_tokens=args.max_tokens,
        extra_body=extra_body,
        response_format=response_format,
    )
    if args.stream:
        for chunk in chat_completion:
            print(chunk.choices[0].delta.content or "", end="")
    else:
        #print(chat_completion.usage)
        print("==== response ====")
        print(chat_completion.choices[0].message.content)
    end_time = time.time()
    print(f"\nTime taken: {end_time - start_time:.2f} seconds, {chat_completion.usage.completion_tokens} tokens")
    print("----------------------------------------------")



async def send_one_request(index, req, args):
    url = args.endpoint + "/chat/completions"
    payload = get_chat_payload(req, args)
    #print(f"==== payload: {payload}")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6 * 60 * 60)) as session:
        request_start_time = time.perf_counter()
        async with session.post(url, headers=OrderedDict({"Content-Type": "application/json"}), json=payload) as res:
            if res.status != 200:
                text = await res.text()
                print(f"ERROR: {res.status}--{res.reason}: {text}")
            else:
                generated = ""
                e2e_latency = 0
                input_tokens = 0
                output_tokens = 0
                async for chunk_bytes in res.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    try:
                        chunk = chunk_bytes.decode("utf-8")
                        if chunk.startswith("data: "):
                            chunk = chunk[6:]
                        #print(f"==== chunk: ++{chunk}++")
                        if chunk == ": OPENROUTER PROCESSING":
                            continue
                        if chunk == "[DONE]":
                            e2e_latency = time.perf_counter() - request_start_time
                        else:
                            obj = json.loads(chunk)
                            #print(f"=== output json: {obj}")
                            content = None
                            if "choices" in obj:
                                choice0 = obj["choices"][0]
                                if "text" in choice0:
                                    content = choice0["text"]
                                elif "delta" in choice0:
                                    if "content" in choice0["delta"]:
                                        content = choice0["delta"]["content"]
                                elif "message" in choice0:
                                    if "content" in choice0["message"]:
                                        content = choice0["message"]["content"]
                            if "usage" in obj:
                                input_tokens = obj["usage"]["prompt_tokens"]
                                output_tokens = obj["usage"]["completion_tokens"]
                            if content is not None:
                                generated += content
                    except json.decoder.JSONDecodeError as err:
                        print(f"JSON DECODE ERROR: {chunk}, {err}")
                    except Exception as err:
                        print(f"Failed to handle streaming chunk: {res.status}, error: {err}")
                if e2e_latency == 0:
                    e2e_latency = time.perf_counter() - request_start_time
                print(f"\n\nTestCase[{index}]\nE2E latency: {e2e_latency:.2f}, sec/token: {(e2e_latency/output_tokens):.3f}, Generated({input_tokens},{output_tokens}):\n{generated}")

async def send_batch_requests(reqs, args):
    num = len(reqs)
    t1 = time.perf_counter()
    print(f"Begin send {num} requests")
    tasks: List[asyncio.Task] = []
    for i in range(num):
        tasks.append(asyncio.create_task(send_one_request(i, reqs[i], args)))
    await asyncio.gather(*tasks)
    t2 = time.perf_counter()
    print(f"End send {num} requests, time: {(t2 - t1):.2f}\n")

def main(args: argparse.Namespace):
    num_prompts = len(USER_PROMPTS)
    base_prompts = USER_PROMPTS * (args.parallel * args.num_tests // num_prompts + 1)
    for i in range(args.num_tests):
        print(f"===== Testing iteration: {i}")
        offset = random.randint(0, len(base_prompts) - args.parallel)
        reqs = base_prompts[offset:offset+args.parallel]
        asyncio.run(send_batch_requests(reqs, args))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSON generation evaluation"
    )
    parser.add_argument("--endpoint", type=str, help="The LLM serving endpoint, for example: http://localhost:18011/v1")
    parser.add_argument("--model", type=str, help="The model name")
    parser.add_argument("--num-tests", type=int, default=1, help="The number of tests, default is 1")
    parser.add_argument("--parallel", type=int, default=1, help="The number of requests at the same time, default is 1")
    parser.add_argument("--max-tokens", type=int, default=1024, help="The max tokens of generated result, default is 1024")
    parser.add_argument("--no-json", action="store_true", help="Disable JSON output if set")
    parser.add_argument("--stream", action="store_true", help="Output with streaming mode")

    args = parser.parse_args()
    main(args)


