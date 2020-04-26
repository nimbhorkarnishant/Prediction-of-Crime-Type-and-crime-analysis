var select = document.getElementById("selectSource");
var select1 = document.getElementById("selectDestination");
var options = [
  "Ahirkhedi",
  "Ajnod",
  "Alok Nagar",
  "Ambedkar Nagar",
  "Anand Nagar",
  "Annapurna",
  "Anoop",
  "NagarAnurag",
  "NagarAshish",
  "NagarAsrawad",
  "KhurdAtahedaAzad",
  "Nagar",
  "Bada Bangarda",
  "Badgounda",
  "Bairathi Colony",
  "Banedia",
  "Banganga",
  "Bangarda Bada",
  "Bangarda Chhota",
  "Banjari",
  "Baoliakhurd",
  "Bardari",
  "Barfani",
  "Barlai Jagir",
  "Bengali Square",
  "Bhagirath Pura",
  "Bhagora",
  "Bhanvarkuan",
  "Bhatkhedi",
  "Bhawrasla",
  "Bhicholi",
  "Bhicholi Hapsi",
  "Bhicholi Mardana",
  "Bijalpur",
  "Brijeshwari Annexe",
  "Budhanya Panth",
  "Budhi Barlai",
  "Bulandshahr",
  "Burankhedi",
  "Chandan",
  "Chhatribagh",
  "Chhatripura",
  "Chhavni",
  "Chhoti Gwaltoli",
  "Chikitsak Nagar",
  "Dakachya",
  "Datoda",
  "DDU Nagar",
  "Depalpur",
  "Dewas Naka",
  "Dudhia",
  "Dwarkapuri",
  "Gadi Adda",
  "Gandhi Nagar",
  "Gautampura",
  "Gawli Palasia",
  "Geeta Nagar",
  "Ghatabillod",
  "Girdhar Nagar",
  "Goyal Nagar",
  "Goyal Vihar",
  "Greater Brijeshwari Annexe",
  "Gumasta Nagar",
  "Harnya Khedi",
  "Harsola",
  "Hasalpur",
  "Hatod",
  "IDA Scheme 140",
  "Indira Gandhi Nagar",
  "Indra Puri Colony",
  "Industrial Estate",
  "Jabran Colony",
  "Jamli",
  "Jhalaria",
  "Joshi Guradiya",
  "Juni",
  "Kachhalya",
  "Kalani Nagar",
  "Kalindi Kunj",
  "Kalindi Mid Town",
  "Kalod Hala",
  "Lalaram Nagar",
  "Lasudia",
  "Lasudia Mori",
  "LIG Colony",
  "Limbodi",
  "Lokmanya Nagar",
  "Maanavta Nagar",
  "Machal",
  "Machla",
  "Mahalaxmi Nagar",
  "Maksi",
  "Malharganj",
  "Malwa Mill",
  "Manavta Nagar",
  "Manbhavan Nagar",
  "Manglia",
  "Nagar Nigam",
  "Nagda",
  "Nainod",
  "Nanda Nagar",
  "Navlakha",
  "New Palasia",
  "New Rani Bagh",
  "Nihalpur",
  "Nihalpur Mandi",
  "Nipania",
  "Niranjanpur",
  "Old Palasia",
  "Pachore",
  "Pagnispaga",
  "Palakhedi",
  "Palda",
  "Paliya Haidar",
  "Palsikar Colony",
  "Panchderiya",
  "Paraspar Nagar",
  "Pardesi Pura",
  "Patni Pura",
  "Pedmi",
  "Raj Mohalla",
  "Rajendra Nagar",
  "Rajmahal Colony",
  "Rajwada",
  "Rala mandal",
  "Rambag",
  "Rangwasa",
  "Rau",
  "Ravi Shankar Shukla Nagar",
  "RSS Nagar",
  "Sadar Bazar",
  "Sai Kripa Colony",
  "Sainath Colony",
  "Saket Nagar",
  "Sanawadia",
  "Sanchar Nagar",
  "Sangam Nagar",
  "Santer",
  "Sanvid Nagar",
  "Sanwer",
  "Sarafa",
  "Sarangpur",
  "Sarv Suvidha Nagar",
  "Sarvsampanna Nagar",
  "Scheme No 103",
  "Scheme No 114",
  "Scheme No 134",
  "Scheme No 140",
  "Scheme No 51",
  "Scheme No 54",
  "Scheme No 71",
  "Scheme No 94",
  "Semliya Chau",
  "Shiv Nagar",
  "Shivaji Nagar",
  "Shivmoti Nagar",
  "Shiwni",
  "Shramik Colony",
  "Shri Nagar",
  "Shri Nagar Extension",
  "Shri Ram Nagar",
  "Shri Ram Talawali",
  "Silicon City",
  "Silver Park Colony",
  "Simrol",
  "Sinhasa",
  "Sirpur",
  "Siyaganj",
  "Smruti Nagar",
  "Sneh Lata Ganj",
  "Sneh Nagar",
  "South Tukoganj",
  "Subhash Nagar",
  "Sudama Nagar",
  "Sukliya",
  "Sula Khedi",
  "Surya Dev Nagar",
  "Vaibhav Nagar",
  "Vaishali Nagar",
  "Vallabh Nagar",
  "Vandana Nagar",
  "Veer Sawarkar Nagar",
  "Vidur Nagar",
  "Vijay Nagar",
  "Vindhyanchal Nagar",
  "Vishnupuri Colony",
  "Talawali Chanda",
  "Tejpur Gadbadi",
  "Telephone Nagar",
  "Tilak Nagar",
  "Tillor Buzurg",
  "Tillor Khurd",
  "Triveni Colony",
  "Tukoganj",
  "Tulsi Nagar",
  "White Church Colony"
];
for (var i = 0; i < options.length; i++) {
  var opt = options[i];
  var el = document.createElement("option");
  el.textContent = opt;
  el.value = opt;
  select.appendChild(el);
}
for (var i = 0; i < options.length; i++) {
  var opt = options[i];
  var el = document.createElement("option");
  el.textContent = opt;
  el.value = opt;
  select1.appendChild(el);
}
