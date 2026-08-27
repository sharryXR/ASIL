/* ASIL project page interactions. Vanilla JavaScript, no dependencies. */
(function () {
  'use strict';

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function setUseIcon(button, iconId) {
    if (!button) return;
    var use = button.querySelector('use');
    if (use) use.setAttribute('href', '#' + iconId);
  }

  (function themeToggle() {
    var root = document.documentElement;
    var button = document.getElementById('themeToggle');
    var stored = null;

    try { stored = localStorage.getItem('asil-theme'); } catch (error) {}

    if (stored === 'light' || stored === 'dark') {
      root.setAttribute('data-theme', stored);
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      root.setAttribute('data-theme', 'dark');
    }

    function syncButton() {
      if (!button) return;
      var isDark = root.getAttribute('data-theme') === 'dark';
      button.setAttribute('aria-label', isDark ? 'Use light theme' : 'Use dark theme');
      button.setAttribute('title', isDark ? 'Use light theme' : 'Use dark theme');
      setUseIcon(button, isDark ? 'icon-sun' : 'icon-moon');
    }

    syncButton();
    if (!button) return;

    button.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('asil-theme', next); } catch (error) {}
      syncButton();
    });
  })();

  (function mobileNavigation() {
    var button = document.getElementById('navBurger');
    var links = document.getElementById('navLinks');
    if (!button || !links) return;

    function close() {
      links.classList.remove('open');
      button.setAttribute('aria-expanded', 'false');
      setUseIcon(button, 'icon-menu');
    }

    button.addEventListener('click', function () {
      var open = links.classList.toggle('open');
      button.setAttribute('aria-expanded', open ? 'true' : 'false');
      setUseIcon(button, open ? 'icon-close' : 'icon-menu');
    });

    links.addEventListener('click', function (event) {
      if (event.target.closest('a')) close();
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') close();
    });
  })();

  (function scrollSpy() {
    var links = Array.prototype.slice.call(document.querySelectorAll('.nav-links a[href^="#"]'));
    if (!links.length || !('IntersectionObserver' in window)) return;

    var targets = [];
    links.forEach(function (link) {
      var target = document.querySelector(link.getAttribute('href'));
      if (target) targets.push({ link: link, target: target });
    });

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (link) { link.classList.remove('active'); });
        var match = targets.find(function (item) { return item.target === entry.target; });
        if (match) match.link.classList.add('active');
      });
    }, { rootMargin: '-42% 0px -52% 0px', threshold: 0 });

    targets.forEach(function (item) { observer.observe(item.target); });
  })();

  (function revealSections() {
    var elements = Array.prototype.slice.call(document.querySelectorAll('.reveal'));
    if (reduceMotion || !('IntersectionObserver' in window)) {
      elements.forEach(function (element) { element.classList.add('in'); });
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('in');
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.1 });

    elements.forEach(function (element) { observer.observe(element); });
  })();

  (function lightbox() {
    var root = document.getElementById('lightbox');
    if (!root) return;

    var image = root.querySelector('img');
    var closeButton = root.querySelector('.close');
    var returnFocus = null;

    function open(source, alt, trigger) {
      returnFocus = trigger;
      image.setAttribute('src', source);
      image.setAttribute('alt', alt || '');
      root.hidden = false;
      root.classList.add('open');
      document.body.classList.add('lightbox-open');
      closeButton.focus();
    }

    function close() {
      if (!root.classList.contains('open')) return;
      root.classList.remove('open');
      root.hidden = true;
      image.removeAttribute('src');
      document.body.classList.remove('lightbox-open');
      if (returnFocus && typeof returnFocus.focus === 'function') returnFocus.focus();
    }

    document.addEventListener('click', function (event) {
      var trigger = event.target.closest('[data-zoomable], .zoomable');
      if (!trigger) return;
      var targetImage = trigger.tagName === 'IMG' ? trigger : trigger.querySelector('img');
      if (!targetImage) return;
      open(targetImage.currentSrc || targetImage.getAttribute('src'), targetImage.getAttribute('alt'), trigger);
    });

    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      var trigger = event.target.closest('[data-zoomable]');
      if (!trigger) return;
      var targetImage = trigger.querySelector('img');
      if (!targetImage) return;
      event.preventDefault();
      open(targetImage.currentSrc || targetImage.getAttribute('src'), targetImage.getAttribute('alt'), trigger);
    });

    root.addEventListener('click', function (event) {
      if (event.target === root || event.target.closest('.close')) close();
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') close();
    });
  })();

  (function copyCitation() {
    var button = document.getElementById('copyBib');
    var pre = document.getElementById('bibText');
    if (!button || !pre) return;

    function done() {
      button.textContent = 'Copied';
      button.classList.add('ok');
      window.setTimeout(function () {
        button.textContent = 'Copy';
        button.classList.remove('ok');
      }, 1600);
    }

    function fallback(text) {
      var textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      try { document.execCommand('copy'); done(); } catch (error) {}
      document.body.removeChild(textarea);
    }

    button.addEventListener('click', function () {
      var text = pre.innerText.trim();
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(text); });
      } else {
        fallback(text);
      }
    });
  })();

  (function synchronizedComparison() {
    var root = document.querySelector('.comparison-loop');
    if (!root) return;

    var tabs = Array.prototype.slice.call(root.querySelectorAll('.dot'));
    var panels = Array.prototype.slice.call(root.querySelectorAll('.panel'));
    var control = document.getElementById('loopPlayPause');
    var count = panels.length;
    var index = 0;
    var timer = null;
    var playing = !reduceMotion;
    var visible = false;
    var dwell = [4600, 4300, 4300, 5200];

    function clearTimer() { window.clearTimeout(timer); }

    function syncControl() {
      if (!control) return;
      control.textContent = playing ? 'Pause' : 'Play';
      control.setAttribute('aria-label', playing ? 'Pause comparison' : 'Play comparison');
    }

    function show(next) {
      index = (next + count) % count;
      root.setAttribute('data-step', String(index));
      panels.forEach(function (panel, panelIndex) {
        var active = panelIndex === index;
        panel.hidden = !active;
        panel.classList.toggle('is-active', active);
      });
      tabs.forEach(function (tab, tabIndex) {
        var active = tabIndex === index;
        tab.setAttribute('aria-selected', active ? 'true' : 'false');
        tab.setAttribute('tabindex', active ? '0' : '-1');
      });
    }

    function schedule() {
      clearTimer();
      if (!playing || !visible) return;
      timer = window.setTimeout(function () {
        show(index + 1);
        schedule();
      }, dwell[index]);
    }

    function pause() {
      playing = false;
      clearTimer();
      syncControl();
    }

    function play() {
      playing = true;
      syncControl();
      schedule();
    }

    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        pause();
        show(Number(tab.getAttribute('data-step')));
      });
      tab.addEventListener('keydown', function (event) {
        var current = tabs.indexOf(tab);
        var target = null;
        if (event.key === 'ArrowRight') target = (current + 1) % count;
        if (event.key === 'ArrowLeft') target = (current - 1 + count) % count;
        if (target === null) return;
        event.preventDefault();
        pause();
        show(target);
        tabs[target].focus();
      });
    });

    if (control) {
      control.addEventListener('click', function () { playing ? pause() : play(); });
    }

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          visible = entry.isIntersecting;
          if (visible) schedule(); else clearTimer();
        });
      }, { threshold: 0.28 }).observe(root);
    } else {
      visible = true;
    }

    document.addEventListener('visibilitychange', function () {
      if (document.hidden) clearTimer(); else schedule();
    });

    show(0);
    syncControl();
    schedule();
  })();
})();
